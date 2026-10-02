"""
Conversation memory and per-session usage limits.

Two jobs, both about protecting a shared GPU from unbounded growth.

Memory: a follow-up like "what about corporate customers?" is meaningless
without the previous turn. History is kept server-side rather than trusted from
the client, because the client could otherwise send an arbitrarily long history
and turn one request into a very expensive prefill.

Limits: every turn carried in the prompt costs prefill for all 50 concurrent
users, and history grows without bound inside one conversation. Capping turns
per session and asking the user to start a new chat keeps the per-request cost
flat no matter how long someone stays. That is what makes 50-user latency
predictable rather than degrading over the day.

State is in-process and deliberately so: it is disposable, it must be fast to
read on the request path, and a restart losing chat history is acceptable.
Running more than one uvicorn worker would split it, which is why WEB_WORKERS
stays at 1.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

import config as cfg


class Session:
    __slots__ = ("id", "turns", "created", "last_seen", "total_questions")

    def __init__(self, sid: str):
        self.id = sid
        self.turns: list[dict] = []
        self.created = time.time()
        self.last_seen = self.created
        self.total_questions = 0

    def history(self) -> list[dict]:
        return self.turns

    def last_documents(self) -> list[str]:
        """Documents cited by the most recent turn that cited anything."""
        for turn in reversed(self.turns):
            if turn.get("documents"):
                return turn["documents"]
        return []

    def last_exchange(self) -> dict | None:
        """The most recent turn, with the sources its citations point at."""
        return self.turns[-1] if self.turns else None

    def add(self, question: str, answer: str, sources: list[dict] | None = None):
        # The documents a turn cited are kept so a follow-up can fall back to
        # them. "Explain more about it" should never come back empty when there
        # is an obvious "it" one turn above.
        docs = []
        for s in (sources or []):
            src = s.get("source")
            if src and src not in docs:
                docs.append(src)
        # Only the newest turn keeps its full source list. It is what a request
        # to shorten the answer is rebuilt from, so its citation numbers can
        # still be resolved to documents -- and older turns would otherwise hold
        # a few hundred dictionaries per conversation that nothing ever reads.
        for turn in self.turns:
            turn.pop("sources", None)
        self.turns.append({"question": question, "answer": answer,
                           "documents": docs[:3], "sources": list(sources or []),
                           "ts": time.time()})
        # Only the recent tail is ever sent to the model, so older turns are dead
        # weight in memory. Keep a little more than the prompt uses so the UI can
        # still show context.
        if len(self.turns) > cfg.SESSION_KEEP_TURNS:
            self.turns = self.turns[-cfg.SESSION_KEEP_TURNS:]
        self.total_questions += 1
        self.last_seen = time.time()

    def remaining(self) -> int:
        return max(0, cfg.SESSION_MAX_QUESTIONS - self.total_questions)

    def is_full(self) -> bool:
        return self.total_questions >= cfg.SESSION_MAX_QUESTIONS

    def info(self) -> dict:
        return {
            "session_id": self.id,
            "questions_used": self.total_questions,
            "questions_limit": cfg.SESSION_MAX_QUESTIONS,
            "questions_remaining": self.remaining(),
            "limit_reached": self.is_full(),
        }


class SessionStore:
    """LRU + TTL over sessions. Both bounds matter: without them a long-running
    server accumulates one entry per browser tab that ever connected."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def get_or_create(self, sid: str | None) -> Session:
        now = time.time()
        with self._lock:
            if sid and sid in self._sessions:
                sess = self._sessions[sid]
                if now - sess.last_seen <= cfg.SESSION_TTL_S:
                    sess.last_seen = now
                    self._sessions.move_to_end(sid)
                    return sess
                del self._sessions[sid]        # expired

            sess = Session(sid or uuid.uuid4().hex)
            self._sessions[sess.id] = sess
            self._sessions.move_to_end(sess.id)
            self._evict_locked(now)
            return sess

    def reset(self, sid: str) -> Session:
        """Start a fresh conversation, reusing nothing from the old one."""
        with self._lock:
            self._sessions.pop(sid, None)
        return self.get_or_create(None)

    def _evict_locked(self, now: float):
        for key in [k for k, s in self._sessions.items()
                    if now - s.last_seen > cfg.SESSION_TTL_S]:
            del self._sessions[key]
        while len(self._sessions) > cfg.SESSION_MAX:
            self._sessions.popitem(last=False)     # oldest touched

    def stats(self) -> dict:
        with self._lock:
            n = len(self._sessions)
            full = sum(1 for s in self._sessions.values() if s.is_full())
        return {
            "active": n,
            "capacity": cfg.SESSION_MAX,
            "at_limit": full,
            "max_questions_per_session": cfg.SESSION_MAX_QUESTIONS,
            "ttl_s": cfg.SESSION_TTL_S,
        }


sessions = SessionStore()

LIMIT_MESSAGE = (
    "This conversation has reached its length limit. Please start a new chat to "
    "continue — your previous messages remain visible in the sidebar."
)
