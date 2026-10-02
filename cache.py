"""
Two-tier answer cache.

L1 -- exact match on the normalised question. The index is deterministic, so an
identical question retrieves identical sources; the cached answer can be served
without touching retrieval or the GPU. This is the sub-10 ms path.

L2 -- semantic match. Cosine similarity over cached question embeddings catches
rephrasings ("what is the LTV cap" vs "what's the maximum loan to value ratio").
OFF by default; config.py carries the measurement that motivates that.

A pure semantic cache is unsafe for compliance. Two questions can be worded
almost identically and still be governed by opposite rules, and the embedding
scores them *closer* than a genuine rephrasing does. So an L2 hit has to clear
three gates, not one:

  1. cosine similarity above CACHE_SIM_THRESHOLD
  2. the fresh retrieval agrees with the retrieval that produced the cached
     answer -- the grounded-cache-routing check from the RAG caching literature
  3. no polarity term flipped between the two questions

Gate 3 exists because gates 1 and 2 both pass for "maximum LTV ratio" vs
"minimum LTV ratio": near-identical wording, the same retrieved chunk, and the
opposite answer.
"""
from __future__ import annotations

import re
import threading
import time

import numpy as np

import config as cfg

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

# Terms that invert or bound the meaning of a compliance question. If two
# questions disagree on any of these groups, they are different questions no
# matter how close their embeddings are.
_POLARITY_GROUPS = [
    {"maximum", "max", "highest", "upper", "ceiling", "cap", "most", "not exceed", "exceeding"},
    {"minimum", "min", "lowest", "floor", "least", "at least"},
    {"shall", "must", "required", "mandatory", "obliged"},
    {"shall not", "must not", "prohibited", "may not", "cannot", "not permitted", "exempt", "excluded"},
    {"before", "prior", "advance", "pre"},
    {"after", "following", "subsequent", "post"},
    {"increase", "increased", "higher", "more"},
    {"decrease", "decreased", "lower", "less", "reduced"},
    {"include", "included", "including", "applies", "applicable", "eligible"},
    {"exclude", "excluded", "excluding", "does not apply", "ineligible"},
]


def normalise(question: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", question.lower())).strip()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def polarity_conflict(a: str, b: str) -> bool:
    """
    True if the two questions differ on any polarity group -- one asks about a
    maximum and the other a minimum, one about what is required and the other
    about what is prohibited, and so on.

    Negations live in their own group rather than as exceptions to the positive
    one, so "shall not" matching both the "shall" group and the "shall not"
    group is harmless: the negative group still differs, which is the conflict.
    """
    a, b = f" {normalise(a)} ", f" {normalise(b)} "
    for group in _POLARITY_GROUPS:
        in_a = any(f" {normalise(t)} " in a for t in group)
        in_b = any(f" {normalise(t)} " in b for t in group)
        if in_a != in_b:
            return True
    return False


class AnswerCache:
    def __init__(self,
                 capacity: int = cfg.CACHE_MAX_ENTRIES,
                 ttl_s: int = cfg.CACHE_TTL_S,
                 dim: int = cfg.EMBED_DIM):
        self.capacity = capacity
        self.ttl_s    = ttl_s
        self._lock    = threading.Lock()

        # Slot-based storage so the embedding matrix stays a single contiguous
        # array -- one matmul scores the whole cache.
        self._vecs    = np.zeros((capacity, dim), dtype=np.float32)
        self._live    = np.zeros(capacity, dtype=bool)
        self._entries: dict[int, dict] = {}
        self._by_key : dict[str, int]  = {}
        self._free   : list[int]       = list(range(capacity))
        self._lru    : list[int]       = []      # slots, oldest first

        self.hits_exact    = 0
        self.hits_semantic = 0
        self.misses        = 0
        self.rejected          = 0   # semantic match whose evidence disagreed
        self.rejected_polarity = 0   # semantic match that flipped a polarity term

    # ── lookup ────────────────────────────────────────────────────────────────
    def get_exact(self, question: str) -> dict | None:
        if not cfg.CACHE_ENABLED:
            return None
        key = normalise(question)
        with self._lock:
            slot = self._by_key.get(key)
            if slot is None:
                return None
            entry = self._entries.get(slot)
            if entry is None or self._expired(entry):
                self._drop(slot)
                return None
            self._touch(slot)
            self.hits_exact += 1
            return dict(entry["payload"], cache="exact")

    def get_semantic(self, question: str, vec: np.ndarray, doc_ids: set[str]) -> dict | None:
        """
        `doc_ids` is the chunk-id set the *new* question just retrieved. The hit
        is only returned if it clears all three gates described at the top of
        this module.
        """
        if not (cfg.CACHE_ENABLED and cfg.CACHE_SEMANTIC_ENABLED):
            self.misses += 1
            return None
        with self._lock:
            if not self._live.any():
                self.misses += 1
                return None
            live_slots = np.flatnonzero(self._live)
            sims = self._vecs[live_slots] @ vec.astype(np.float32)
            best = int(np.argmax(sims))
            if sims[best] < cfg.CACHE_SIM_THRESHOLD:
                self.misses += 1
                return None

            slot  = int(live_slots[best])
            entry = self._entries.get(slot)
            if entry is None or self._expired(entry):
                self._drop(slot)
                self.misses += 1
                return None

            if _jaccard(entry["doc_ids"], doc_ids) < cfg.CACHE_MIN_DOC_OVERLAP:
                # Wording matched, evidence did not. Generate a fresh answer.
                self.rejected += 1
                self.misses   += 1
                return None

            if polarity_conflict(question, entry["key"]):
                # Same words, same sources, opposite question. This is the case
                # similarity alone cannot catch.
                self.rejected_polarity += 1
                self.misses += 1
                return None

            self._touch(slot)
            self.hits_semantic += 1
            return dict(entry["payload"],
                        cache="semantic",
                        cache_similarity=round(float(sims[best]), 4))

    # ── insert ────────────────────────────────────────────────────────────────
    def put(self, question: str, vec: np.ndarray, doc_ids: set[str], payload: dict):
        if not cfg.CACHE_ENABLED:
            return
        key = normalise(question)
        with self._lock:
            slot = self._by_key.get(key)
            if slot is None:
                if not self._free:
                    self._evict_one()
                slot = self._free.pop()
            self._vecs[slot] = vec.astype(np.float32)
            self._live[slot] = True
            self._by_key[key] = slot
            self._entries[slot] = {
                "key": key, "doc_ids": set(doc_ids),
                "payload": payload, "ts": time.time(),
            }
            self._touch(slot)

    # ── housekeeping (callers already hold the lock) ──────────────────────────
    def _expired(self, entry: dict) -> bool:
        return self.ttl_s > 0 and (time.time() - entry["ts"]) > self.ttl_s

    def _touch(self, slot: int):
        try:
            self._lru.remove(slot)
        except ValueError:
            pass
        self._lru.append(slot)

    def _evict_one(self):
        if self._lru:
            self._drop(self._lru[0])

    def _drop(self, slot: int):
        entry = self._entries.pop(slot, None)
        if entry is not None:
            self._by_key.pop(entry["key"], None)
        self._live[slot] = False
        try:
            self._lru.remove(slot)
        except ValueError:
            pass
        if slot not in self._free:
            self._free.append(slot)

    def clear(self):
        with self._lock:
            self._live[:] = False
            self._entries.clear()
            self._by_key.clear()
            self._lru.clear()
            self._free = list(range(self.capacity))

    def stats(self) -> dict:
        total = self.hits_exact + self.hits_semantic + self.misses
        return {
            "enabled"       : cfg.CACHE_ENABLED,
            "entries"       : int(self._live.sum()),
            "capacity"      : self.capacity,
            "hits_exact"    : self.hits_exact,
            "hits_semantic" : self.hits_semantic,
            "misses"        : self.misses,
            "semantic_enabled"   : cfg.CACHE_SEMANTIC_ENABLED,
            "rejected_ungrounded": self.rejected,
            "rejected_polarity"  : self.rejected_polarity,
            "hit_rate"      : round((self.hits_exact + self.hits_semantic) / total, 4) if total else 0.0,
        }


answer_cache = AnswerCache()
