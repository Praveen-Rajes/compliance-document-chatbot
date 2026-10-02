#!/usr/bin/env python3
"""
Compliance RAG serving API.

The request path is fully async end to end. Nothing blocks the event loop except
work that genuinely releases the GIL (ONNX embedding, the BLAS search, PyMuPDF),
and that runs on a small executor.

Latency tiers, fastest first:

  identity question    no retrieval, no search    answered from corpus facts
  exact cache hit      ~5 ms                      no retrieval, no GPU
  generation           TTFT-bound, streamed token by token

    python server.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import fitz          # PyMuPDF, for live citation highlighting
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config as cfg
import prompts
from cache import answer_cache
from llm import llm
from retriever import retriever
from queryfix import (GREETING_REPLY, PROFANITY_REPLY, TOO_SHORT_REPLY, fixer,
                      is_greeting, is_offensive, set_profanity_level)
from sessions import LIMIT_MESSAGE, sessions
import usagelog

log = logging.getLogger("server")

_STATIC_DIR = Path(__file__).parent / "static"
# A citation bracket may hold several numbers: the model writes [1, 2, 3] as
# readily as [1]. An earlier version matched only a single digit inside
# brackets, so a thorough answer citing [1, 2, 3] parsed as zero citations and
# showed no sources at all -- and with no sources there was nothing for the
# document viewer to open, which looked like the highlighter being broken.
_CITE_BLOCK_RE = re.compile(r"\[\s*(\d+(?:\s*[,;/&]\s*\d+)*)\s*\]")
_CITE_NUM_RE   = re.compile(r"\d+")
_QUOTE_RE      = re.compile(r'"([^"]{3,300})"\s*((?:\[\s*[\d,;/&\s]+\s*\])+)')


def _cited_numbers(text: str) -> list[int]:
    """Every citation number in order of first appearance, flattening [1, 2, 3]."""
    out, seen = [], set()
    for block in _CITE_BLOCK_RE.findall(text):
        for n in _CITE_NUM_RE.findall(block):
            i = int(n)
            if i not in seen:
                seen.add(i)
                out.append(i)
    return out

_admission: asyncio.Semaphore | None = None

_stats = {
    "start_time": 0.0, "total": 0, "failed": 0, "rejected": 0,
    "below_thresh": 0, "by_kind": {}, "corrected": 0,
    "expanded": 0, "rejected_junk": 0, "followup_fallbacks": 0,
    "latency_ms": deque(maxlen=2000), "ttft_ms": deque(maxlen=2000),
}


# ── citation parsing ──────────────────────────────────────────────────────────
_EMPTY_CITE_RE = re.compile(r"\s*\[\s*\]")

# Bracketed text that is not a citation.
#
# Sources are numbered plainly -- [1], [2] -- but the model also copies clause
# numbering straight out of the document it is quoting and writes it the same
# way: [4.1], [9.1.a], [6.6.c, d, e, f, g, h], [5.j]. Those point at a section of
# a PDF, not at a source, so there is nothing for a click to open. On screen they
# were indistinguishable from a citation that had failed, which is what made the
# citations as a whole look unreliable. Every bracket that is not a citation is
# removed, so nothing that looks clickable is dead.
_BRACKETED_RE          = re.compile(r"[ \t]*\[[^\[\]\n]{0,120}\]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([.,;:!?)])")
_RUN_OF_SPACES_RE      = re.compile(r"[ \t]{2,}")


def _drop_stray_brackets(text: str) -> str:
    """Remove every bracket that is not a source citation."""
    return _BRACKETED_RE.sub(
        lambda m: m.group(0) if _CITE_BLOCK_RE.fullmatch(m.group(0).strip()) else "",
        text)


def _tidy_spacing(text: str) -> str:
    """Close the gap left behind where a bracket was removed."""
    return _RUN_OF_SPACES_RE.sub(" ", _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text))


#: A sentence ends at .!? followed by a capital, an opening bracket or a
#: bullet. Requiring the capital is what keeps "Circular No. 2 of 2019" and
#: "Rs. 500" in one piece, since a digit follows the full stop there.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\u2022])|\n+")

#: An explicit quotation inside a sentence. When the model quotes the source
#: verbatim, that is a better highlight target than the sentence around it.
_QUOTED_SPAN_RE = re.compile(r'"([^"]{3,300})"')

#: Below this a sentence carries too few words to locate reliably on the page,
#: and a confident highlight in the wrong place is worse than none.
_MIN_QUOTE_CHARS = 25


def _citation_marks(text: str, sources: list[dict]) -> list[dict]:
    """
    One highlight target per citation *occurrence*, in the order they are read.

    A citation number identifies a document, so an answer drawing eight facts
    out of one circular writes [1] eight times. The source list holds one entry
    per number, carrying one quote -- so every one of those eight markers opened
    the PDF on the same sentence, the first one. Only the first citation ever
    highlighted what it was actually attached to.

    Each marker is therefore paired with the sentence it closes, which is the
    claim it is evidence for. The page and document still come from the cited
    source; only the highlighted text differs between occurrences.
    """
    marks, floor, carried = [], 0, None
    for m in _CITE_BLOCK_RE.finditer(text):
        i = int(m.group(1)) - 1
        segment, floor = text[floor:m.start()], m.end()
        if not (0 <= i < len(sources)):
            continue

        # Back up to the start of the sentence this marker closes.
        bounds = list(_SENTENCE_END_RE.finditer(segment))
        if bounds:
            segment = segment[bounds[-1].end():]
        segment = _CITE_BLOCK_RE.sub("", segment).replace("**", "")
        segment = segment.strip(" \t\n*-\u2013\u2014:;,")

        # Prefer a verbatim quotation over the prose around it.
        quoted = _QUOTED_SPAN_RE.search(segment)
        if quoted:
            segment = quoted.group(1).strip()

        if len(segment) >= _MIN_QUOTE_CHARS:
            carried = segment[:300]
        quote = carried or sources[i].get("quote")
        marks.append({**sources[i], "quote": quote})
    return marks


def build_citations(answer: str, used: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """
    Renumber the citations in an answer so they line up with the sources returned.

    The model numbers its citations against the sources it was *given*, so an
    answer might cite [2] and [5] out of five supplied passages. Only cited
    sources are shown to the reader, which compacts that to a two-item list --
    and the marker [5] then points at a fourth entry that does not exist, while
    [2] points at the wrong one. In the interface that showed up as some
    citations opening the wrong document and others sitting there as dead text
    in square brackets.

    So the markers are rewritten to match: [2] becomes [1], [5] becomes [2],
    and the returned list is in exactly that order. Markers pointing outside the
    supplied sources are dropped, as is a bare "[]".

    Returns the rewritten answer, the sources it now refers to, and one
    highlight target per marker -- see _citation_marks.
    """
    quotes: dict[int, str] = {}
    for quote, tags in _QUOTE_RE.findall(answer):
        for n in _CITE_NUM_RE.findall(tags):
            quotes.setdefault(int(n), quote.strip())

    # Order of first appearance is the order the reader meets them in.
    cited = [n for n in _cited_numbers(answer) if 1 <= n <= len(used)]
    remap = {old: new for new, old in enumerate(cited, start=1)}

    sources = []
    for old in cited:
        h = used[old - 1]
        sources.append({
            "filename": h["filename"], "folder": h["folder"],
            "source": h["source"], "page": h["page"],
            "score": h["score"], "quote": quotes.get(old),
        })

    def fix(m):
        nums = [int(x) for x in _CITE_NUM_RE.findall(m.group(1))]
        return "".join(f"[{remap[i]}]" for i in nums if i in remap)

    # Order matters: renumber first, so the surviving markers are plain integers,
    # then take out everything else that is in brackets.
    text = _CITE_BLOCK_RE.sub(fix, _EMPTY_CITE_RE.sub("", answer))
    text = _tidy_spacing(_drop_stray_brackets(text)).strip()
    return text, sources, _citation_marks(text, sources)


# ── core pipeline ─────────────────────────────────────────────────────────────
async def prepare(question: str, mode: str, session, min_score: float):
    """
    Everything up to calling the LLM. Returns either a finished answer (cache hit
    or below threshold) or the messages to generate from.
    """
    t0 = time.perf_counter()
    history = session.history() if session else []

    # Offensive language is refused before anything else, so no retrieval runs
    # and no GPU time is spent on a message that will not be answered.
    if cfg.PROFANITY_REFUSE and is_offensive(question):
        _stats["by_kind"]["offensive"] = _stats["by_kind"].get("offensive", 0) + 1
        return {"done": {
            "question": question, "answer": PROFANITY_REPLY, "sources": [],
            "confidence": 0.0, "cache": "miss", "kind": "offensive",
            "timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
        }}

    # Greetings and pleasantries answer instantly with no retrieval and no GPU.
    if is_greeting(question):
        _stats["by_kind"]["greeting"] = _stats["by_kind"].get("greeting", 0) + 1
        return {"done": {
            "question": question, "answer": GREETING_REPLY, "sources": [],
            "confidence": 100.0, "cache": "miss", "kind": "greeting",
            "timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
        }}

    # Spelling is repaired against the corpus vocabulary before anything else,
    # so a typo cannot mis-route the question or poison the embedding.
    fixed_q, corrections = fixer.correct(question)
    kind = prompts.classify(fixed_q)

    # "Make it shorter" is not a new question. It is the answer already on the
    # user's screen, in fewer words -- so it is answered from the last turn and
    # never touches the index. Searching again would return a different short
    # answer rather than a shorter version of the one they are reading, and
    # could cite documents that answer never mentioned.
    #
    # wants_brevity requires that the sentence names no new subject, so
    # "summarise the Mobile Payments Guidelines No. 2 of 2011" stays a normal
    # document summary. Overview and identity questions are excluded outright:
    # "give me a brief overview of what you know" is a question about the
    # library, not a request to shorten anything.
    previous = session.last_exchange() if session else None
    if (previous and previous.get("answer")
            and kind not in ("identity", "overview")
            and fixer.wants_brevity(fixed_q, history)):
        kind = "condense"

    # "auto" for condensing: an explicit mode=detailed would otherwise hand back
    # a thousand tokens to a request whose entire point is fewer words.
    b = prompts.budget(kind, "auto" if kind == "condense" else mode)
    _stats["by_kind"][kind] = _stats["by_kind"].get(kind, 0) + 1
    if corrections:
        _stats["corrected"] += 1
        log.info("query corrected: %s", corrections)

    if kind == "condense":
        log.info("condensing the previous answer (%d chars)",
                 len(previous["answer"]))
        return {
            "messages": prompts.build_condense_messages(question,
                                                        previous["answer"]),
            # The previous turn's sources, in their original order, so a [1]
            # carried through the shortened answer still opens the same page.
            "used": list(previous.get("sources") or []),
            "vec": None, "doc_ids": set(), "confidence": 100.0,
            "retrieve_ms": 0.0, "max_tokens": b["max_tokens"],
            "kind": "condense", "t0": t0, "cacheable": False,
        }

    # Quality gate. Returns the confidence bar this particular question has to
    # clear -- higher for short or unrecognised input, so "s" cannot come back
    # as a confident citation while "LCR" still works.
    q_meta = fixer.quality(fixed_q, min_score)
    if not q_meta["ok"] and kind not in ("identity", "overview"):
        _stats["rejected_junk"] += 1
        return {"done": {
            "question": question, "answer": TOO_SHORT_REPLY, "sources": [],
            "confidence": 0.0, "cache": "miss", "kind": "unclear",
            "timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
        }}
    min_score = q_meta["min_score"]
    # A cached answer was produced without this conversation's context, so it is
    # only safe to reuse when there is no context to miss.
    cacheable = not history

    if cacheable:
        cached = answer_cache.get_exact(question)
        if cached is not None:
            cached["timing"] = {"total_ms": round((time.perf_counter() - t0) * 1000, 1)}
            cached["kind"] = kind
            return {"done": cached}

    # Identity questions never touch the index: the honest answer is a
    # description of the library, which comes from its metadata.
    if kind == "identity":
        return {
            "messages": prompts.build_messages(
                question, "", "identity", history,
                prompts.corpus_brief(retriever.corpus)),
            "used": [], "vec": None, "doc_ids": set(), "confidence": 100.0,
            "retrieve_ms": 0.0, "max_tokens": b["max_tokens"], "kind": kind,
            "t0": t0, "cacheable": cacheable,
        }

    # Follow-ups ("tell me more about it") carry no topic of their own, so the
    # previous turn's content words are folded into the *retrieval* query only.
    search_q, expanded = fixer.expand(fixed_q, history)
    if expanded:
        _stats["expanded"] += 1

    hits, vec = await retriever.retrieve(search_q, b["retrieve"])
    best = (hits[0]["score"] / 100) if hits else 0.0


    # For a follow-up, search the document already under discussion FIRST.
    #
    # Falling back only when the corpus search fails is not enough, because it
    # often does not fail -- it succeeds with the wrong document. "That answer
    # is too long, give me a more concise one" matched an unrelated Customer
    # Charter at 71%, well clear of the confidence bar, so the answer came back
    # correct (the model used the conversation) while citing a document it had
    # never read from. A confident citation to the wrong file is worse than a
    # low-confidence one to the right file.
    fell_back = False
    prev_docs = session.last_documents() if session else []
    if prev_docs and not prompts.is_document_scoped(fixed_q) \
            and fixer.is_followup(fixed_q, history):
        scoped = await retriever.within_documents(prev_docs, vec, b["retrieve"])
        if scoped and (scoped[0]["score"] / 100) >= cfg.FOLLOWUP_SCOPE_FLOOR:
            hits, fell_back = scoped, True
            best = scoped[0]["score"] / 100
            _stats["followup_fallbacks"] += 1
            log.info("follow-up scoped to the document under discussion: %s (%.1f%%)",
                     [d.rsplit("/", 1)[-1] for d in prev_docs], best * 100)
    retrieve_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Overview questions are about the library, not about one passage, so a low
    # top-1 similarity is expected and must not trigger the "not found" path.
    # A question that names a document has already told us which one it means,
    # so the confidence bar is lowered for it: refusing outright when the user
    # was specific is worse than answering from the closest match and citing it.
    if kind == "document":
        min_score = min(min_score, cfg.DOCUMENT_MIN_SCORE)

    if not fell_back and kind != "overview" and best < min_score:
        _stats["below_thresh"] += 1
        return {"done": {
            "question": question, "answer": prompts.NOT_FOUND, "sources": [],
            "confidence": round(best * 100, 1), "cache": "miss", "kind": kind,
            "timing": {"retrieve_ms": retrieve_ms,
                       "total_ms": round((time.perf_counter() - t0) * 1000, 1)},
        }}

    if kind == "document":
        # Everything from the best-matching document, in reading order.
        picked = await retriever.deepen(hits, vec, b["generate"])
    else:
        picked = retriever.diversify(hits, b["generate"], b["per_doc"])
    context, used = prompts.build_context(picked, b["chars"])
    doc_ids = {h["id"] for h in used}

    if cacheable:
        cached = answer_cache.get_semantic(question, vec, doc_ids)
        if cached is not None:
            cached["timing"] = {"retrieve_ms": retrieve_ms,
                                "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
            cached["kind"] = kind
            return {"done": cached}

    corpus_text = prompts.corpus_brief(retriever.corpus) if kind == "overview" else ""
    return {
        "messages": prompts.build_messages(question, context, kind, history, corpus_text),
        "used": used, "vec": vec, "doc_ids": doc_ids,
        "confidence": round(best * 100, 1), "retrieve_ms": retrieve_ms,
        "max_tokens": b["max_tokens"], "kind": kind, "t0": t0,
        "cacheable": cacheable,
    }


#: Appended when the model ran out of room. Without it an answer simply stops
#: mid-sentence, which reads as a wrong answer rather than an incomplete one.
TRUNCATED_NOTE = ("\n\n_This answer reached its length limit and may be "
                  "incomplete. Ask for the remaining items to continue._")


def _finish(question, text, plan, ttft_ms=None, truncated=False):
    text, sources, marks = build_citations(text, plan["used"])
    if truncated and not text.endswith(TRUNCATED_NOTE):
        text += TRUNCATED_NOTE
    total = round((time.perf_counter() - plan["t0"]) * 1000, 1)
    result = {
        "question": question, "answer": text, "sources": sources, "marks": marks,
        "confidence": plan["confidence"], "cache": "miss", "kind": plan["kind"],
        "timing": {"retrieve_ms": plan["retrieve_ms"],
                   "ttft_ms": ttft_ms, "total_ms": total},
    }
    if plan.get("cacheable") and plan.get("vec") is not None:
        answer_cache.put(question, plan["vec"], plan["doc_ids"],
                         {k: result[k] for k in
                          ("question", "answer", "sources", "marks", "confidence",
                           "kind")})
    return result


async def answer_once(question: str, mode: str, session, min_score: float) -> dict:
    plan = await prepare(question, mode, session, min_score)
    if "done" in plan:
        return plan["done"]
    text, usage, finish = await llm.complete(plan["messages"], plan["max_tokens"])
    result = _finish(question, text.strip(), plan, truncated=(finish == "length"))
    result["usage"] = usage
    result["truncated"] = finish == "length"
    return result


def _client_ip(request: Request) -> str:
    """
    The caller's address, for the per-user transcript.

    X-Forwarded-For wins where it is present: behind a reverse proxy the socket
    address is the proxy's, and every user would otherwise share one file. Only
    the first entry is taken -- the rest of that header is the chain of proxies
    -- and the whole thing is client-supplied, which is why usagelog sanitises
    it before it becomes a filename.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def answer_stream(question: str, mode: str, session, min_score: float,
                        client: str = ""):
    """Emits: meta -> token* -> sources -> done."""
    plan = await prepare(question, mode, session, min_score)
    sess_info = session.info() if session else {}

    if "done" in plan:
        d = plan["done"]
        yield _sse("meta", {"confidence": d.get("confidence", 0.0),
                            "cache": d.get("cache", "miss"),
                            "kind": d.get("kind"), "session": sess_info})
        yield _sse("token", {"text": d["answer"]})
        yield _sse("sources", d.get("sources", []))
        if session:
            session.add(question, d["answer"], d.get("sources"))
        await usagelog.record(client, question, d["answer"],
                              kind=d.get("kind", ""),
                              confidence=d.get("confidence", 0.0),
                              sources=d.get("sources"), timing=d.get("timing"),
                              session_id=session.id if session else "")
        yield _sse("done", {"timing": d.get("timing", {}),
                            "cache": d.get("cache", "miss"),
                            "answer": d["answer"], "marks": d.get("marks", []),
                            "session": session.info() if session else {}})
        _stats["total"] += 1
        _stats["latency_ms"].append(d.get("timing", {}).get("total_ms", 0.0))
        return

    yield _sse("meta", {"confidence": plan["confidence"], "cache": "miss",
                        "kind": plan["kind"],
                        "sources_considered": len(plan["used"]),
                        "session": sess_info})

    parts, ttft_ms, usage, finish = [], None, {}, None
    try:
        async for delta, chunk_usage, chunk_finish in llm.stream(
                plan["messages"], plan["max_tokens"]):
            if chunk_finish:
                finish = chunk_finish
            if chunk_usage:
                usage = chunk_usage
                continue
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - plan["t0"]) * 1000, 1)
                _stats["ttft_ms"].append(ttft_ms)
            parts.append(delta)
            yield _sse("token", {"text": delta})
    except Exception as exc:                                        # noqa: BLE001
        _stats["failed"] += 1
        log.exception("generation failed")
        yield _sse("error", {"detail": str(exc)})
        return

    text = "".join(parts).strip()
    result = _finish(question, text, plan, ttft_ms, truncated=(finish == "length"))
    if session:
        # result["answer"], not the raw stream: the citation numbers in it have
        # been renumbered to match the sources, and history is replayed to the
        # model, so storing the raw text would feed back numbering that no
        # longer means anything.
        session.add(question, result["answer"], result["sources"])
    await usagelog.record(client, question, result["answer"],
                          kind=result.get("kind", ""),
                          confidence=result.get("confidence", 0.0),
                          sources=result.get("sources"),
                          timing=result.get("timing"),
                          session_id=session.id if session else "")

    if finish == "length":
        yield _sse("token", {"text": TRUNCATED_NOTE})
    yield _sse("sources", result["sources"])
    # The browser has been accumulating raw tokens, which still carry the
    # model's original numbering. Send the corrected text so the final render
    # uses markers that match the source list.
    yield _sse("done", {"cache": "miss", "usage": usage, "truncated": finish == "length",
                        "answer": result["answer"], "marks": result["marks"],
                        "timing": result["timing"],
                        "session": session.info() if session else {}})
    _stats["total"] += 1
    _stats["latency_ms"].append(result["timing"]["total_ms"])


# ── lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _admission
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    _stats["start_time"] = time.time()
    _admission = asyncio.Semaphore(cfg.MAX_INFLIGHT)
    set_profanity_level(cfg.PROFANITY_MILD)

    log.info("=" * 64)
    log.info("  Compliance RAG -- serving")
    log.info("=" * 64)
    log.info("model         : %s", cfg.LLM_MODEL)
    log.info("vLLM backends : %s", ", ".join(cfg.VLLM_HOSTS))
    log.info("max in-flight : %d", cfg.MAX_INFLIGHT)
    log.info("session cap   : %d questions, %d min TTL",
             cfg.SESSION_MAX_QUESTIONS, cfg.SESSION_TTL_S // 60)
    log.info("language filter: %s (mild terms %s)",
             "on" if cfg.PROFANITY_REFUSE else "off",
             "included" if cfg.PROFANITY_MILD else "allowed")

    await llm.start()
    await llm.wait_ready()
    await asyncio.get_running_loop().run_in_executor(None, retriever.load)

    c = retriever.corpus
    log.info("corpus        : %d documents, %d pages, %d chunks, %d collections",
             c.get("documents", 0), c.get("pages", 0),
             c.get("chunks", 0), len(c.get("folders", [])))

    try:
        await answer_once("warmup", "concise", None, 0.0)
        answer_cache.clear()
        log.info("Warmup complete.")
    except Exception as exc:                                        # noqa: BLE001
        log.warning("Warmup failed (non-fatal): %s", exc)

    log.info("Ready. %d chunks indexed.", retriever.count())
    log.info("UI      http://localhost:%d/", cfg.API_PORT)
    log.info("=" * 64)
    yield
    await llm.close()
    retriever.close()


app = FastAPI(
    title="Amana Compliance RAG API",
    description="High-concurrency compliance RAG over vLLM, with cited answers.",
    version="3.0.0", lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

# Serves anything under static/. The page itself is returned by the "/" route;
# this mount exists for assets. Note there is deliberately no JavaScript library
# here: document highlighting is rendered server-side by PyMuPDF, so the app has
# no third-party front-end code to ship, review or keep patched.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── schemas ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="Omit to start a new conversation.")
    mode: str = Field("auto", pattern="^(auto|concise|balanced|detailed)$")
    min_score: float = Field(cfg.MIN_SCORE, ge=0.0, le=1.0)

    model_config = {"json_schema_extra": {"example": {
        "question": "What is the loan to value ratio for motor vehicle credit facilities?",
        "mode": "auto",
    }}}


class Source(BaseModel):
    filename: str
    folder: str
    source: str
    page: int
    score: float
    quote: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    confidence: float
    cache: str = "miss"
    kind: str = "specific"
    timing: dict
    usage: dict = {}
    session: dict = {}


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Compliance RAG</h1><p>API up. See <a href='/docs'>/docs</a>.</p>")
    # no-store, because a cached copy of this page survives a rebuild and leaves
    # users on old front-end code while the server runs new code -- a confusing
    # failure that looks like a bug in the application.
    return HTMLResponse(index.read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "model": cfg.LLM_MODEL,
            "chunks_indexed": retriever.count(),
            "uptime_s": round(time.time() - _stats["start_time"], 1),
            "backends": cfg.VLLM_HOSTS}


@app.get("/corpus", tags=["system"], summary="What the knowledge base contains")
async def corpus():
    c = dict(retriever.corpus)
    c["folders"] = [{"folder": f, "documents": n} for f, n in c.get("folders", [])]
    return c


def _pct(values, p: float) -> float:
    if not values:
        return 0.0
    o = sorted(values)
    return round(o[min(len(o) - 1, int(len(o) * p))], 1)


@app.get("/stats", tags=["system"])
async def stats():
    lat, ttft = list(_stats["latency_ms"]), list(_stats["ttft_ms"])
    return {
        "uptime_s": round(time.time() - _stats["start_time"], 1),
        "requests_total": _stats["total"], "requests_failed": _stats["failed"],
        "rejected_busy": _stats["rejected"], "below_threshold": _stats["below_thresh"],
        "by_question_kind": _stats["by_kind"],
        "queries_spell_corrected": _stats["corrected"],
        "followups_expanded": _stats["expanded"],
        "followup_fallbacks": _stats["followup_fallbacks"],
        "usage_log": usagelog.summary(),
        "rejected_unclear": _stats["rejected_junk"],
        "latency_ms": {"p50": _pct(lat, .50), "p95": _pct(lat, .95), "p99": _pct(lat, .99)},
        "ttft_ms": {"p50": _pct(ttft, .50), "p95": _pct(ttft, .95), "p99": _pct(ttft, .99)},
        "cache": answer_cache.stats(), "llm": llm.stats(),
        "retrieval": retriever.stats(), "sessions": sessions.stats(),
    }


@app.post("/session/new", tags=["chat"], summary="Start a fresh conversation")
async def new_session(body: dict | None = None):
    old = (body or {}).get("session_id")
    sess = sessions.reset(old) if old else sessions.get_or_create(None)
    return sess.info()


@app.post("/cache/clear", tags=["system"])
async def clear_cache():
    answer_cache.clear()
    return {"cleared": True}


async def _admit():
    try:
        await asyncio.wait_for(_admission.acquire(), timeout=cfg.QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        _stats["rejected"] += 1
        raise HTTPException(503, "Server saturated, retry shortly.",
                            headers={"Retry-After": "2"})


@app.post("/ask", response_model=AskResponse, tags=["chat"])
async def ask(req: AskRequest, request: Request):
    if retriever.count() == 0:
        raise HTTPException(503, "Index is empty. Run ingest.py first.")
    sess = sessions.get_or_create(req.session_id)
    if sess.is_full():
        return AskResponse(question=req.question, answer=LIMIT_MESSAGE, sources=[],
                           confidence=0.0, kind="limit", timing={},
                           session=sess.info())
    await _admit()
    t0 = time.perf_counter()
    try:
        result = await answer_once(req.question, req.mode, sess, req.min_score)
        sess.add(req.question, result["answer"], result.get("sources"))
        await usagelog.record(_client_ip(request), req.question, result["answer"],
                              kind=result.get("kind", ""),
                              confidence=result.get("confidence", 0.0),
                              sources=result.get("sources"),
                              timing=result.get("timing"),
                              session_id=sess.id)
        result["session"] = sess.info()
        _stats["total"] += 1
        _stats["latency_ms"].append(round((time.perf_counter() - t0) * 1000, 1))
        return AskResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:                                        # noqa: BLE001
        _stats["failed"] += 1
        log.exception("ask failed: %s", req.question[:120])
        raise HTTPException(500, str(exc))
    finally:
        _admission.release()


@app.post("/ask/stream", tags=["chat"], summary="Streamed answer (SSE)")
async def ask_stream(req: AskRequest, request: Request):
    if retriever.count() == 0:
        raise HTTPException(503, "Index is empty. Run ingest.py first.")
    sess = sessions.get_or_create(req.session_id)

    if sess.is_full():
        async def limited():
            yield _sse("meta", {"kind": "limit", "session": sess.info()})
            yield _sse("token", {"text": LIMIT_MESSAGE})
            yield _sse("sources", [])
            yield _sse("done", {"timing": {}, "session": sess.info()})
        return StreamingResponse(limited(), media_type="text/event-stream")

    await _admit()
    client = _client_ip(request)

    async def gen():
        try:
            async for chunk in answer_stream(req.question, req.mode, sess,
                                             req.min_score, client):
                yield chunk
        finally:
            _admission.release()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive",
        "X-Accel-Buffering": "no",      # stop nginx buffering the stream
    })


# ── document endpoints ────────────────────────────────────────────────────────
def _safe_resolve(source: str) -> Path:
    """
    `source` arrives from the client on every call, so this containment check is
    what stops a crafted path reading files outside the dataset.
    """
    path = Path(source).resolve()
    try:
        path.relative_to(cfg.DATASET_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied: path outside dataset directory")
    if not path.exists():
        raise HTTPException(404, "PDF not found")
    return path


@app.get("/api/pdf", tags=["documents"])
async def api_pdf(source: str, page: int = 0, highlight: str = ""):
    """
    The original PDF. With `page` and `highlight`, a copy is returned with a real
    highlight annotation drawn on that page, so the browser's own PDF viewer
    shows it without any help from us. The file on disk is never modified.
    """
    path = _safe_resolve(source)
    if not (highlight and page):
        return FileResponse(str(path), media_type="application/pdf",
                            headers={"Cache-Control": "public, max-age=3600"})

    def annotate():
        doc = fitz.open(str(path))
        try:
            found, boxes, _how, _ratio = find_quote(doc, highlight, page)
            if boxes:
                _draw_boxes(doc[found - 1], boxes)
            return doc.tobytes(garbage=3, deflate=True)
        finally:
            doc.close()

    data = await asyncio.get_running_loop().run_in_executor(None, annotate)
    return Response(content=data, media_type="application/pdf", headers={
        "Cache-Control": "private, max-age=600",
        "Content-Disposition": f'inline; filename="{path.name}"',
    })


# --- highlight -----------------------------------------------------------------
_WORD_RE = re.compile(r"[0-9a-z]+")

#: Words too common to anchor the edge of a highlight on. A window that
#: begins or ends on one of these has almost certainly overshot the phrase.
_EDGE_WORDS = frozenset("""
    the a an of and or to in for that this these those is are was were
    be been being by as it its on at with from which who whose such any
    all shall may must not no than then there their have has had
""".split())


def _norm_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Drops the punctuation, quote style,
    ligatures and hyphenation that make exact matching fail."""
    return _WORD_RE.findall(text.lower().replace("’", "'").replace("–", "-"))


def _match_words(page, quote: str, min_ratio: float = 0.55):
    """
    Locate a quote by matching token runs against the page's word boxes.

    PyMuPDF's search_for is an exact substring match on the text layer, which
    fails constantly here for reasons that have nothing to do with the model
    being wrong: the indexed chunk may come from vision transcription rather than
    the text layer, PDFs hyphenate across lines, and ligatures, curly quotes and
    non-breaking spaces all differ from what was extracted. Matching normalised
    token runs instead tolerates every one of those.

    Returns (boxes, ratio). Boxes are merged per line so a phrase spanning two
    lines produces two rectangles rather than one covering the gap.
    """
    want = _norm_tokens(quote)
    if not want:
        return [], 0.0

    words = page.get_text("words")          # x0,y0,x1,y1,word,block,line,word_no
    if not words:
        return [], 0.0                      # scanned page, no text layer

    norm, index = [], []
    for i, w in enumerate(words):
        toks = _norm_tokens(w[4])
        for t in toks:
            norm.append(t)
            index.append(i)
    if not norm:
        return [], 0.0

    n = len(want)
    best_ratio, best_at = 0.0, -1
    want_set = set(want)
    # Slide a window the length of the quote and score by token overlap. Cheap
    # enough at page scale, and robust to a word or two differing.
    for start in range(0, max(1, len(norm) - n + 1)):
        window = norm[start:start + n]
        if not window:
            break
        hits = sum(1 for a, b in zip(window, want) if a == b)
        if hits < n * 0.4:
            # cheap reject before the more forgiving set-overlap score
            loose = len(want_set.intersection(window))
            score = loose / n * 0.8
        else:
            score = hits / n
        if score > best_ratio:
            best_ratio, best_at = score, start
            if best_ratio >= 0.99:
                break

    if best_at < 0 or best_ratio < min_ratio:
        return [], best_ratio

    # The window is as long as the quote, so wording the page does not share --
    # a citation now carries the model's whole sentence, which opens with its own
    # framing -- pushes the window's edges past the text that actually matched.
    # Pull them in to the outermost content words the quote and the page share,
    # so the highlight covers the sentence rather than a fixed-length span
    # drifting off one end of it. This can only shrink the span, never move it.
    lo, hi = best_at, min(best_at + n - 1, len(norm) - 1)
    while lo < hi and (norm[lo] not in want_set or norm[lo] in _EDGE_WORDS):
        lo += 1
    while hi > lo and (norm[hi] not in want_set or norm[hi] in _EDGE_WORDS):
        hi -= 1

    first_w = index[lo]
    last_w = index[min(hi, len(index) - 1)]
    chosen = words[first_w:last_w + 1]

    by_line: dict[tuple, list] = {}
    for w in chosen:
        by_line.setdefault((w[5], w[6]), []).append(w)
    boxes = []
    for group in by_line.values():
        x0 = min(g[0] for g in group); y0 = min(g[1] for g in group)
        x1 = max(g[2] for g in group); y1 = max(g[3] for g in group)
        boxes.append([x0, y0, x1, y1])
    return boxes, best_ratio


def _locate(pg, text: str):
    """
    Find a quote on an open page. Four strategies, cheapest first.

    Returns (boxes, how, ratio) where boxes are fitz.Rect-compatible lists in
    PDF coordinates.
    """
    quote = " ".join(text.split())

    # 1. exact, then whitespace-normalised
    for probe in (text.strip(), quote):
        if probe:
            r = pg.search_for(probe)
            if r:
                return [list(x) for x in r], "exact", 1.0

    # 2. leading word windows, longest first -- a long quote often fails only
    #    because of its tail
    toks = quote.split()
    for take in (14, 10, 8, 6, 5):
        if len(toks) > take:
            r = pg.search_for(" ".join(toks[:take]))
            if r:
                return [list(x) for x in r], "partial", round(take / len(toks), 3)

    # 3. normalised token-run match, which survives hyphenation and OCR
    boxes, ratio = _match_words(pg, quote)
    if boxes:
        return boxes, "fuzzy", round(ratio, 3)

    return [], ("no_match" if pg.get_text("words") else "no_text_layer"), 0.0


#: Yellow used for the highlight, matching the colour the UI used to draw.
_HL_COLOUR = (1.0, 0.85, 0.10)

#: Above this, a rendered page is sent as JPEG instead of PNG. Text pages land
#: well under it; scanned pages are photographs and blow past it.
_IMG_PNG_LIMIT = 400_000


#: How far from the cited page to keep looking, in pages. A citation names a
#: retrieved passage, and the sentence the model attached to it is very often a
#: page or two away -- the model draws a long answer out of one document and
#: cites the same passage throughout. Bounded so a several-hundred-page PDF
#: cannot turn one click into a full-document scan.
_PAGE_SEARCH_RADIUS = 30


def _page_order(hint: int, total: int):
    """Page numbers to try, nearest the cited page first."""
    hint = min(max(hint, 1), total)
    yield hint
    for step in range(1, min(_PAGE_SEARCH_RADIUS, total) + 1):
        for n in (hint - step, hint + step):
            if 1 <= n <= total:
                yield n


def _confidence(how: str, ratio: float) -> float:
    """
    Put the three matching strategies on one scale, so pages can be compared.

    _locate reports a ratio, but it means something different in each strategy:
    an exact hit is always 1.0, a partial hit reports how much of the quote
    matched verbatim, and a fuzzy hit reports token overlap. Comparing those
    numbers directly would rank a fourteen-word verbatim run (0.42) below a
    mediocre fuzzy match somewhere else in the document (0.60).

    A verbatim run is direct evidence the text is on that page, so it starts
    from a floor and rises with its length; a fuzzy match is inference, and is
    trusted at face value. The two meet where they should: fourteen verbatim
    words score about the same as a 0.7 fuzzy match.
    """
    if how == "exact":
        return 1.0
    if how == "partial":
        return 0.5 + 0.5 * ratio
    return ratio


def find_quote(doc, text: str, hint: int) -> tuple[int, list, str, float]:
    """
    Locate a quote in an open document. Returns (page, boxes, how, ratio).

    Searching only the cited page was leaving later citations unhighlighted. A
    citation number identifies a retrieved passage, which sits on one page --
    but an answer drawing a dozen points out of one circular attaches them all
    to that passage, and most of those points are written on other pages. The
    document opened at the right place with nothing marked on it.

    Pages are tried nearest-first and the best match across them wins, rather
    than the first one over the bar: the page that really carries the sentence
    scores far above a coincidental match elsewhere, so comparing is what keeps
    the highlight honest. An exact hit stops the search immediately.
    """
    best, best_conf = (hint, [], "no_match", 0.0), 0.0
    for n in _page_order(hint, len(doc)):
        boxes, how, ratio = _locate(doc[n - 1], text)
        if not boxes:
            continue
        conf = _confidence(how, ratio)
        if conf > best_conf:
            best, best_conf = (n, boxes, how, ratio), conf
            if how == "exact":
                break
    return best


def _draw_boxes(pg, boxes) -> None:
    """
    Draw the highlight annotations on a page.

    The page has to be a live object the caller is holding, not a fresh
    `doc[n]` fetched per box. An annotation belongs to the page it came from,
    and indexing the document hands back a new page object each time -- so the
    one the annotation was created on is collected almost immediately, and
    colouring it then fails with "annotation not bound to any page". Taking the
    page as an argument keeps it referenced for the whole loop.
    """
    for box in boxes:
        annot = pg.add_highlight_annot(fitz.Rect(box))
        annot.set_colors(stroke=_HL_COLOUR)
        annot.set_opacity(0.45)
        annot.update()


def _apply_highlight(pg, text: str) -> str:
    """Draw a real PDF highlight annotation over the quote. Returns the match kind."""
    boxes, how, _ = _locate(pg, text)
    _draw_boxes(pg, boxes)
    return how


@app.get("/api/highlight", tags=["documents"],
         summary="Locate a cited quote on a page (diagnostics)")
async def api_highlight(source: str, page: int, text: str):
    path = _safe_resolve(source)

    def find():
        doc = fitz.open(str(path))
        try:
            if not (1 <= page <= len(doc)):
                raise HTTPException(400, f"Page {page} out of range (1-{len(doc)})")
            pg = doc[page - 1]
            boxes, how, ratio = _locate(pg, text)
            return boxes, pg.rect.width, pg.rect.height, how, ratio
        finally:
            doc.close()

    boxes, w, h, how, ratio = await asyncio.get_running_loop().run_in_executor(None, find)
    return {"boxes": boxes, "page_width": w, "page_height": h,
            "match": how, "ratio": ratio}


@app.get("/api/locate", tags=["documents"],
         summary="Which page a cited sentence is actually on")
async def api_locate(source: str, page: int = 1, text: str = ""):
    """
    Resolve a citation to the page that carries its sentence.

    The viewer needs this before it opens the file: the page to scroll to is
    part of the URL, so it has to be known up front. Called once per click,
    never on the answer path.
    """
    path = _safe_resolve(source)
    if not text:
        return {"page": page, "match": "none", "ratio": 0.0}

    def find():
        doc = fitz.open(str(path))
        try:
            found, boxes, how, ratio = find_quote(doc, text, page)
            return found, bool(boxes), how, ratio
        finally:
            doc.close()

    found, hit, how, ratio = await asyncio.get_running_loop().run_in_executor(None, find)
    return {"page": found if hit else page, "match": how, "ratio": ratio}


@app.get("/api/page_image", tags=["documents"],
         summary="One page rendered as an image, with the quote highlighted")
async def api_page_image(source: str, page: int, text: str = "", zoom: float = 2.0):
    """
    The cited page as a PNG with the highlight already drawn on it.

    This is what the document viewer shows. Rendering server-side rather than in
    the browser means no PDF JavaScript library has to be shipped -- which
    removes 1.4 MB of third-party minified code from the deployment, and with it
    the only component we could not read line by line. PyMuPDF is already here
    for ingestion, so this costs no new dependency.

    It is also more reliable: the same code that located the quote draws the
    box, so the highlight cannot drift out of alignment the way a separately
    rendered overlay can.
    """
    path = _safe_resolve(source)
    zoom = max(1.0, min(zoom, 3.0))

    def render():
        doc = fitz.open(str(path))
        try:
            if not (1 <= page <= len(doc)):
                raise HTTPException(400, f"Page {page} out of range (1-{len(doc)})")
            pg = doc[page - 1]
            how = _apply_highlight(pg, text) if text else "none"
            pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

            # PNG is the right format for a text page: sharp and small, because
            # most of the page is flat white. A scanned page is a photograph and
            # compresses terribly as PNG -- one measured here at 877 KB, versus
            # about a tenth of that as JPEG with no visible difference. Rather
            # than guess from the page type, render PNG and switch if it comes
            # out heavy.
            png = pix.tobytes("png")
            if len(png) <= _IMG_PNG_LIMIT:
                return png, "image/png", len(doc), how
            return pix.tobytes("jpeg", jpg_quality=82), "image/jpeg", len(doc), how
        finally:
            doc.close()

    data, mime, total, how = await asyncio.get_running_loop().run_in_executor(None, render)
    return Response(content=data, media_type=mime, headers={
        "Cache-Control": "private, max-age=600",
        "X-Total-Pages": str(total),
        "X-Highlight-Match": how,
    })


if __name__ == "__main__":
    uvicorn.run("server:app", host=cfg.API_HOST, port=cfg.API_PORT,
                workers=cfg.WEB_WORKERS, log_level="info",
                access_log=False,        # per-request logging is a real cost at 50 rps
                timeout_keep_alive=75)
