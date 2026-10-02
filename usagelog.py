"""
A readable transcript per user, one file each, under logs/.

Every answered question is appended to a file named after the client's address,
so `logs/10.20.30.41.log` is the complete history of what that person asked and
what they were told. The format is meant to be opened in Notepad and read, not
parsed: a dated header line, the question, the answer as it was shown, and the
documents it was drawn from.

Two things this deliberately does not do. It never raises: a failure to write a
log line must not cost the user their answer, so every path here swallows its
own errors after warning once. And it never blocks the event loop -- the write
goes to a worker thread, because fifty people asking at once would otherwise
queue up behind each other's disk I/O.

The address is taken from X-Forwarded-For when the application sits behind a
proxy, and from the socket otherwise. Where users share an outbound address --
NAT, a VPN concentrator, a single office gateway -- their entries land in one
file, because the application has no login to tell them apart.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from datetime import datetime
from pathlib import Path

import config as cfg

log = logging.getLogger("usagelog")

#: Anything that is not plainly safe in a filename becomes an underscore. This
#: covers IPv6 colons on Windows, and stops a spoofed X-Forwarded-For header
#: from walking out of the log directory.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

_RULE = "=" * 78
_THIN = "-" * 78

#: One lock per file. Fifty users write concurrently, and appends from separate
#: threads can otherwise interleave mid-line.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

_warned = False


def path_for(client: str) -> Path:
    """The log file for one client address."""
    name = _UNSAFE.sub("_", (client or "").strip()).strip("._-")
    return Path(cfg.LOG_DIR) / f"{(name or 'unknown')[:60]}.log"


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(str(path), threading.Lock())


def _render(question: str, answer: str, kind: str, confidence: float,
            sources: list[dict] | None, timing: dict | None,
            session_id: str) -> str:
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    head = [when]
    if kind:
        head.append(kind)
    if confidence:
        head.append(f"confidence {confidence:.1f}%")
    total = (timing or {}).get("total_ms")
    if total:
        head.append(f"{total / 1000:.1f}s")
    if session_id:
        head.append(f"chat {session_id[:8]}")

    lines = [_RULE, "  |  ".join(head), _THIN,
             "QUESTION", (question or "").strip(), "",
             "ANSWER", (answer or "").strip()]

    # One line per distinct document, in citation order, so the entry says what
    # the answer was actually built from.
    if sources:
        seen, listed = set(), []
        for s in sources:
            key = s.get("source") or s.get("filename")
            if key in seen:
                continue
            seen.add(key)
            page = s.get("page")
            listed.append(f"  [{len(listed) + 1}] {s.get('filename', '?')}"
                          + (f"  page {page}" if page else ""))
        if listed:
            lines += ["", "SOURCES"] + listed

    return "\n".join(lines) + "\n\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_for(path):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)


async def record(client: str, question: str, answer: str, *,
                 kind: str = "", confidence: float = 0.0,
                 sources: list[dict] | None = None,
                 timing: dict | None = None,
                 session_id: str = "") -> None:
    """Append one exchange to this client's transcript. Never raises."""
    global _warned
    if not cfg.LOG_USAGE:
        return
    try:
        text = _render(question, answer, kind, confidence, sources, timing,
                       session_id)
        await asyncio.get_running_loop().run_in_executor(
            None, _write, path_for(client), text)
    except Exception as exc:                                        # noqa: BLE001
        if not _warned:
            _warned = True
            log.warning("usage logging is failing and will be skipped: %s", exc)


def summary() -> dict:
    """What the health endpoint reports about logging."""
    directory = Path(cfg.LOG_DIR)
    try:
        files = sorted(directory.glob("*.log"))
        return {"enabled": cfg.LOG_USAGE, "directory": str(directory),
                "users": len(files),
                "bytes": sum(f.stat().st_size for f in files)}
    except Exception:                                               # noqa: BLE001
        return {"enabled": cfg.LOG_USAGE, "directory": str(directory),
                "users": 0, "bytes": 0}
