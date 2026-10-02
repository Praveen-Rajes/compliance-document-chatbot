"""
Retrieval: vector search + a micro-batched query embedder.

Why not ChromaDB. Chroma's PersistentClient does a SQLite metadata join on every
query and holds the GIL through most of it, so 50 concurrent /ask calls serialise
in Python before they ever reach the GPU. Here metadata is a plain in-memory
list -- lookup is an array index, not a database round trip.

Why exact search by default. At this corpus size (roughly 40k chunks from ~1000
PDFs) a float32 matmul against the whole matrix takes ~1 ms, releases the GIL
inside BLAS, and has perfect recall. HNSW only starts paying for itself past
~150k chunks, where it is used automatically if an index was built. Below that
it would trade recall away for nothing.

Why micro-batch the embedder. One ONNX forward for 32 short queries costs barely
more than one forward for a single query, so arrivals are collected in a
few-millisecond window and embedded together. That turns 50 sequential CPU
forwards into one or two.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config as cfg

log = logging.getLogger(__name__)


# ── Micro-batched query embedder ──────────────────────────────────────────────
class EmbedBatcher:
    """
    Collects concurrent embed requests and runs the group through fastembed as
    one batch on a worker thread.

    The window is adaptive. Waiting a fixed few milliseconds would be pure added
    latency for a lone request on an idle server, but skipping the wait entirely
    would never batch anything under load. So: if no encode is currently
    running, flush immediately; if one is, wait the window, because those
    milliseconds would have been spent queued behind it regardless and the
    arrivals that land meanwhile ride along for free.
    """

    def __init__(self, model, executor: ThreadPoolExecutor):
        self._model     = model
        self._executor  = executor
        self._pending: list[tuple[str, asyncio.Future]] = []
        self._lock      = asyncio.Lock()
        self._scheduled = False
        self._running   = 0
        self.batches    = 0
        self.embedded   = 0

    async def embed(self, query: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        async with self._lock:
            self._pending.append((query, fut))
            immediate = (self._running == 0
                         or len(self._pending) >= cfg.EMBED_BATCH_MAX)
            if not self._scheduled:
                self._scheduled = True
                loop.create_task(
                    self._flush(0.0 if immediate else cfg.EMBED_BATCH_WINDOW_MS / 1000)
                )
        return await fut

    async def _flush(self, delay: float):
        if delay:
            await asyncio.sleep(delay)
        async with self._lock:
            batch, self._pending = self._pending, []
            self._scheduled = False
            self._running += 1
        if not batch:
            async with self._lock:
                self._running -= 1
            return

        loop = asyncio.get_running_loop()
        try:
            vecs = await loop.run_in_executor(
                self._executor, self._encode, [q for q, _ in batch]
            )
        except Exception as exc:                                    # noqa: BLE001
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return
        finally:
            async with self._lock:
                self._running -= 1

        self.batches  += 1
        self.embedded += len(batch)
        for (_, fut), vec in zip(batch, vecs):
            if not fut.done():
                fut.set_result(vec)

    def _encode(self, texts: list[str]) -> np.ndarray:
        # query_embed applies the BGE retrieval instruction prefix; embed() does
        # not. Using the wrong one silently costs several points of recall.
        vecs = np.asarray(list(self._model.query_embed(texts)), dtype=np.float32)
        return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)

    def stats(self) -> dict:
        return {
            "batches"  : self.batches,
            "embedded" : self.embedded,
            "avg_batch": round(self.embedded / self.batches, 2) if self.batches else 0.0,
        }


# ── Vector index ──────────────────────────────────────────────────────────────
class Retriever:
    """Holds the vector matrix, the chunk metadata, and the embedding model."""

    HNSW_THRESHOLD = 150_000

    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._meta: list[dict] = []
        self._hnsw = None
        self._model = None
        self._executor: ThreadPoolExecutor | None = None
        self.batcher: EmbedBatcher | None = None
        self.backend = "none"
        self.corpus: dict = {}
        self._by_source: dict = {}

    # -- lifecycle --
    def load(self):
        from fastembed import TextEmbedding

        if not cfg.VECTORS_PATH.exists():
            raise RuntimeError(f"No index at {cfg.VECTORS_PATH}. Run `python ingest.py` first.")

        t0 = time.time()
        with cfg.META_PATH.open("r", encoding="utf-8") as fh:
            self._meta = [json.loads(line) for line in fh]

        # mmap keeps startup instant and lets several uvicorn workers share one
        # copy of the matrix through the page cache instead of holding one each.
        self._vectors = np.load(cfg.VECTORS_PATH, mmap_mode="r")
        if len(self._meta) != len(self._vectors):
            raise RuntimeError(
                f"Index mismatch: {len(self._meta)} metadata rows vs "
                f"{len(self._vectors)} vectors. Re-run ingest.py."
            )

        # Chunk indices grouped by document, so every chunk of one file can be
        # gathered without scanning the whole index.
        self._by_source = {}
        for i, m in enumerate(self._meta):
            self._by_source.setdefault(m.get("source"), []).append(i)

        self.corpus = self._corpus_stats()

        from queryfix import fixer
        n_vocab = fixer.build(self._meta)
        log.info("Spelling vocabulary: %d words from the corpus", n_vocab)

        self.backend = "exact"
        if len(self._meta) >= self.HNSW_THRESHOLD and cfg.INDEX_PATH.exists():
            try:
                import hnswlib
                idx = hnswlib.Index(space="cosine", dim=int(self._vectors.shape[1]))
                idx.load_index(str(cfg.INDEX_PATH), max_elements=len(self._meta))
                idx.set_ef(cfg.HNSW_EF_SEARCH)
                self._hnsw = idx
                self.backend = "hnsw"
            except Exception as exc:                                # noqa: BLE001
                log.warning("HNSW index unusable (%s) -- falling back to exact search", exc)

        log.info("Index loaded: %d chunks, dim=%d, backend=%s (%.1fs)",
                 len(self._meta), self._vectors.shape[1], self.backend, time.time() - t0)

        self._executor = ThreadPoolExecutor(max_workers=cfg.EMBED_THREADS,
                                            thread_name_prefix="embed")
        # Offline loading is enforced by HF_HUB_OFFLINE, set in config.py --
        # fastembed 0.8 has no local_files_only argument, so passing one here
        # would be silently ignored.
        self._model = TextEmbedding(
            model_name=cfg.EMBED_MODEL,
            cache_dir=cfg.EMBED_CACHE,
            threads=1,          # parallelism comes from the executor, not from
        )                       # intra-op threads competing with each other
        list(self._model.query_embed(["warmup"]))
        self.batcher = EmbedBatcher(self._model, self._executor)
        log.info("Embedding model ready: %s", cfg.EMBED_MODEL)

    def close(self):
        if self._executor:
            self._executor.shutdown(wait=False)

    def _corpus_stats(self) -> dict:
        """
        Describe the indexed library from its own metadata.

        This exists so "what do you know?" can be answered from facts rather than
        from whatever three chunks a vector search happens to return. Cheap to
        build (one pass over metadata already in memory) and it means the numbers
        quoted to a user are always true of the index actually loaded.
        """
        from collections import Counter, defaultdict
        import re as _re

        folder_docs = defaultdict(set)
        pages_by_doc = {}
        years = set()
        for m in self._meta:
            doc = m.get("source") or m.get("filename", "")
            folder = m.get("folder") or "(root)"
            folder_docs[folder].add(doc)
            pages_by_doc[doc] = max(pages_by_doc.get(doc, 0), int(m.get("total_pages") or 1))
            for y in _re.findall(r"\b(19|20)\d{2}\b", folder + " " + m.get("filename", "")):
                pass
            ym = _re.search(r"\b(19\d{2}|20\d{2})\b", folder)
            if ym:
                years.add(int(ym.group(1)))

        folders = sorted(((f, len(d)) for f, d in folder_docs.items()),
                         key=lambda kv: -kv[1])
        # Longest filenames are the most descriptive, so they make the best
        # examples of what the library actually holds.
        titles = sorted({m.get("filename", "") for m in self._meta}, key=len, reverse=True)
        sample = [t[:110] for t in titles[:12]]

        return {
            "documents": len(pages_by_doc),
            "pages": sum(pages_by_doc.values()),
            "chunks": len(self._meta),
            "folders": folders,
            "years": sorted(years),
            "sample_titles": sample,
        }

    # -- queries --
    def count(self) -> int:
        return len(self._meta)

    async def retrieve(self, question: str, top_k: int) -> tuple[list[dict], np.ndarray]:
        vec  = await self.batcher.embed(question)
        hits = await self.search(vec, top_k)
        return hits, vec

    async def deepen(self, hits: list[dict], vec, limit: int) -> list[dict]:
        """
        Pull the most relevant chunks from the single best-matching document.

        This is the opposite of diversify(), and both are needed. "What are the
        requirements for banks?" wants breadth across documents. "What are the
        definitions in General Direction No. 01 of 2013?" wants everything from
        one document -- and the per-document cap that gives the first question
        good coverage is exactly what truncates the second.

        Measured on this corpus: that direction holds 21 definitions spread over
        6 chunks. Capped at 2 chunks per document, only 3 definitions could ever
        be reported, however the question was worded.
        """
        if not hits:
            return hits
        source = hits[0].get("source")
        idxs = self._by_source.get(source)
        if not idxs:
            return hits[:limit]

        loop = asyncio.get_running_loop()
        order = await loop.run_in_executor(self._executor, self._rank_within, idxs, vec)
        picked = [{**self._meta[i], "score": round(float(sc) * 100, 1)}
                  for i, sc in order[:limit]]
        # Reading order is what a list of definitions needs; relevance decided
        # which chunks, not how they are presented.
        picked.sort(key=lambda h: h["id"])
        return picked

    async def within_documents(self, sources: list[str], vec, limit: int) -> list[dict]:
        """
        Best chunks from a named set of documents, ignoring the rest of the index.

        Used as the safety net for follow-up questions: if "explain more about
        it" scores too low against the whole corpus to be answerable, searching
        only the document the previous answer came from is far better than
        telling the user nothing was found.
        """
        idxs = [i for src in sources for i in self._by_source.get(src, [])]
        if not idxs:
            return []
        loop = asyncio.get_running_loop()
        order = await loop.run_in_executor(self._executor, self._rank_within, idxs, vec)
        return [{**self._meta[i], "score": round(float(sc) * 100, 1)}
                for i, sc in order[:limit]]

    def _rank_within(self, idxs, vec):
        sims = self._vectors[idxs] @ vec
        pairs = sorted(zip(idxs, sims), key=lambda t: -t[1])
        return pairs

    @staticmethod
    def diversify(hits: list[dict], limit: int, per_doc: int) -> list[dict]:
        """
        Take the best `limit` chunks while allowing at most `per_doc` from any one
        document.

        Without this a broad question routinely fills every slot with consecutive
        chunks of a single PDF -- the retriever is working correctly, they really
        are the nearest neighbours, but the answer then cites one document and
        looks like the system only knows about one document. Capping per document
        trades a little top-1 similarity for coverage, which is what a question
        like "what are the requirements" actually wants.

        Relevance order is preserved; nothing is re-ranked, only skipped.
        """
        if per_doc <= 0:
            return hits[:limit]
        seen, out, overflow = {}, [], []
        for h in hits:
            key = h.get("source") or h.get("filename")
            if seen.get(key, 0) < per_doc:
                seen[key] = seen.get(key, 0) + 1
                out.append(h)
                if len(out) >= limit:
                    return out
            else:
                overflow.append(h)
        # Backfill from the skipped chunks if the cap left us short of `limit`.
        out.extend(overflow[: limit - len(out)])
        return out

    async def search(self, vec: np.ndarray, top_k: int) -> list[dict]:
        k = min(top_k, len(self._meta))
        if k == 0:
            return []
        loop = asyncio.get_running_loop()
        idxs, sims = await loop.run_in_executor(self._executor, self._knn, vec, k)
        return [{**self._meta[int(i)], "score": round(float(s) * 100, 1)}
                for i, s in zip(idxs, sims)]

    def _knn(self, vec: np.ndarray, k: int):
        """Runs on a worker thread. Both branches release the GIL internally."""
        if self._hnsw is not None:
            labels, dists = self._hnsw.knn_query(vec.reshape(1, -1), k=k)
            return labels[0], 1.0 - dists[0]

        # Both sides are L2-normalised, so the dot product is cosine similarity.
        sims = self._vectors @ vec
        if k >= len(sims):
            top = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
        return top, sims[top]

    def stats(self) -> dict:
        return {
            "backend"  : self.backend,
            "chunks"   : len(self._meta),
            "documents": self.corpus.get("documents", 0) if self.corpus else 0,
            "pages"    : self.corpus.get("pages", 0) if self.corpus else 0,
            "folders"  : len(self.corpus.get("folders", [])) if self.corpus else 0,
            "embed"    : self.batcher.stats() if self.batcher else {},
        }


retriever = Retriever()
