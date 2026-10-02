"""
Central configuration. Every value is overridable by environment variable so the
same image runs unchanged on the RTX 4500 dev box and the 2x L40S server.
"""
import os
from pathlib import Path


def _int(name, default):   return int(os.getenv(name, str(default)))
def _float(name, default): return float(os.getenv(name, str(default)))
def _bool(name, default):  return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")
def _path(name, default):  return Path(os.getenv(name, default))


# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR   = _path("DATASET_DIR",   "./dataset")
INDEX_DIR     = _path("INDEX_DIR",     "./index")   # bind-mounted folder
IMAGES_DIR    = _path("IMAGES_DIR",    "./page_images")
PROGRESS_FILE = _path("PROGRESS_FILE", "./index/ingest_progress.json")

# One readable transcript per user, named after their address. Bind-mount this
# directory in docker-compose.yml or the transcripts live inside the container
# and are destroyed the next time the image is replaced.
LOG_DIR       = _path("LOG_DIR",       "./logs")
LOG_USAGE     = _bool("LOG_USAGE",     True)

VECTORS_PATH  = INDEX_DIR / "vectors.npy"   # always written; exact search reads this
INDEX_PATH    = INDEX_DIR / "hnsw.bin"      # only built for very large corpora
META_PATH     = INDEX_DIR / "meta.jsonl"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
# Extracted + vision-transcribed page text, written before the embedding
# stage so a failure there never costs another OCR pass. See --from-pages.
PAGES_PATH    = INDEX_DIR / "pages.jsonl"

# ── Models ────────────────────────────────────────────────────────────────────
LLM_MODEL   = os.getenv("LLM_MODEL",   "google/gemma-4-E2B-it")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM   = _int("EMBED_DIM", 384)
EMBED_CACHE = os.getenv("EMBED_CACHE_DIR", "./embedding_cache")

# Force fastembed to use the baked-in cache and never reach HuggingFace.
#
# This has to be done with environment variables, not a constructor argument.
# fastembed 0.8's TextEmbedding takes no `local_files_only` parameter -- passing
# one lands in **kwargs and is silently ignored, after which fastembed compares
# the cached files against HuggingFace metadata, decides they do not match, and
# blocks trying to re-download. On an air-gapped box that hangs forever with no
# error. HF_HUB_OFFLINE makes huggingface_hub serve straight from the cache.
#
# Set here, at import time, because every entry point imports config before
# anything imports fastembed. Works inside Docker and on bare metal alike.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ── vLLM backends ─────────────────────────────────────────────────────────────
# Comma-separated so prod can list one replica per GPU:
#   VLLM_HOSTS=http://vllm-0:8000,http://vllm-1:8000
VLLM_HOSTS   = [h.strip().rstrip("/") for h in
                os.getenv("VLLM_HOSTS", os.getenv("VLLM_HOST", "http://127.0.0.1:8000")).split(",")
                if h.strip()]
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
VLLM_TIMEOUT = _float("VLLM_TIMEOUT", 90.0)
VLLM_CONNECT_TIMEOUT = _float("VLLM_CONNECT_TIMEOUT", 5.0)

# ── Ingestion ─────────────────────────────────────────────────────────────────
IMAGE_DPI        = _int("IMAGE_DPI",        150)
MIN_TEXT_LEN     = _int("MIN_TEXT_LEN",     80)
USE_VISION       = _bool("USE_VISION",      True)
INGEST_WORKERS   = _int("INGEST_WORKERS",   8)
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 256)
VISION_CONCURRENCY = _int("VISION_CONCURRENCY", 8)
# Page text longer than this is split into overlapping chunks so a single hit
# never drags a 6000-token page into the prompt.
CHUNK_CHARS   = _int("CHUNK_CHARS",   1400)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)

# HNSW is only built and used above ~150k chunks (see Retriever.HNSW_THRESHOLD).
# Below that, exact search is ~1 ms with perfect recall, so approximation buys
# nothing. ef_search is the latency/recall dial when HNSW is in play.
HNSW_M               = _int("HNSW_M",               32)
HNSW_EF_CONSTRUCTION = _int("HNSW_EF_CONSTRUCTION", 200)
HNSW_EF_SEARCH       = _int("HNSW_EF_SEARCH",       96)

# ── Sessions ──────────────────────────────────────────────────────────────────
# Every turn of history is prefill paid for on every subsequent request, by all
# 50 concurrent users. Capping the conversation keeps per-request cost flat
# instead of growing all day, and the UI asks the user to start a new chat when
# the cap is hit. Raise SESSION_MAX_QUESTIONS if the GPU has headroom.
#
# These are defaults. docker-compose.yml sets SESSION_MAX_QUESTIONS in the
# environment, and the environment wins -- so the compose file is the place to
# change it, and a restart applies it, not a rebuild.
SESSION_MAX_QUESTIONS = _int("SESSION_MAX_QUESTIONS", 20)
# Kept equal to the question limit on purpose: a chat is capped at 20 questions,
# and every one of them stays available to the model, so the last question of a
# chat can still refer back to the first. Older answers are trimmed by length
# rather than dropped, which is what keeps the prefill cost bounded.
SESSION_KEEP_TURNS    = _int("SESSION_KEEP_TURNS", 20)
SESSION_TTL_S         = _int("SESSION_TTL_S", 2 * 3600)
SESSION_MAX           = _int("SESSION_MAX", 2000)

# ── Retrieval ─────────────────────────────────────────────────────────────────
# These are floors. The per-question budget in prompts.BUDGETS is what actually
# drives retrieval width -- an overview question pulls 60 candidates and sends 8
# chunks from 8 different documents, a specific one pulls 24 and sends 4. A
# single fixed TOP_K_GENERATE=3 was why broad questions only ever cited two or
# three PDFs.
TOP_K_RETRIEVE = _int("TOP_K_RETRIEVE", 24)
TOP_K_GENERATE = _int("TOP_K_GENERATE", 4)
# Raised from 0.28. Junk input ("s", "ssss") was matching real documents at
# 56-70% and producing confident cited answers, while genuine questions score
# 77-95%. queryfix.quality() raises this further for very short or
# unrecognised queries, so acronyms like LCR and KYC still work.
MIN_SCORE      = _float("MIN_SCORE",    0.45)
# Hard cap on characters of retrieved context handed to the model. This is the
# main TTFT dial: prefill cost is linear in it.
MAX_CONTEXT_CHARS = _int("MAX_CONTEXT_CHARS", 4200)

# ── Generation ────────────────────────────────────────────────────────────────
# 110 tokens at ~64 tok/s/user (batch 50, RTX 4500) plus ~2x from n-gram
# speculative decoding lands a complete answer just under 1 s. Raise to 320 for
# the "detailed" mode, which trades that for a fuller answer.
# Global ceilings applied on top of the per-question-type budgets in
# prompts.BUDGETS. Set either to 0 to leave the routed budget alone.
#
# These exist so answer length and prompt size can be tuned from the compose
# file without editing Python. They CLAMP the routed budget, never raise it:
# ANSWER_TOKEN_CAP=250 makes every answer at most 250 tokens, which is the
# quickest way to trade thoroughness for speed across the whole system.
ANSWER_TOKEN_CAP   = _int("ANSWER_TOKEN_CAP",   0)
CONTEXT_CHAR_CAP   = _int("CONTEXT_CHAR_CAP",   0)

MAX_TOKENS         = _int("MAX_TOKENS",         160)
MAX_TOKENS_CONCISE = _int("MAX_TOKENS_CONCISE", 110)
MAX_TOKENS_DETAIL  = _int("MAX_TOKENS_DETAIL",  384)
TEMPERATURE        = _float("TEMPERATURE", 0.2)

# ── Caching ───────────────────────────────────────────────────────────────────
# L1, exact match on the normalised question. Provably safe: identical text
# retrieves identical sources, so the cached answer is the answer. Always on.
# Offensive language is refused outright, including when the underlying question
# is a valid one. Set PROFANITY_MILD=false to allow milder words ("damn",
# "hell") through while still refusing strong ones.
PROFANITY_REFUSE = _bool("PROFANITY_REFUSE", True)
PROFANITY_MILD   = _bool("PROFANITY_MILD",   True)

# When a follow-up continues the previous turn, the document already under
# discussion is searched first. This is the score that scoped search must reach
# before it is preferred over a fresh search of the whole corpus. Below it, the
# previous document probably does not cover what was asked.
FOLLOWUP_SCOPE_FLOOR = _float("FOLLOWUP_SCOPE_FLOOR", 0.50)

# Confidence bar for a question that names a document. Lower than the general
# one: the user has already said which document they mean, so answering from the
# closest match beats refusing.
DOCUMENT_MIN_SCORE   = _float("DOCUMENT_MIN_SCORE", 0.30)

CACHE_ENABLED       = _bool("CACHE_ENABLED", True)
CACHE_MAX_ENTRIES   = _int("CACHE_MAX_ENTRIES", 4096)
CACHE_TTL_S         = _int("CACHE_TTL_S", 6 * 3600)

# L2, semantic match. OFF by default, deliberately.
#
# Measured on this corpus with bge-small-en-v1.5:
#     "maximum LTV ratio for motor vehicles"  vs
#     "minimum LTV ratio for motor vehicles"        cosine 0.9434
#     "maximum LTV ratio for motor vehicles"  vs
#     "highest LTV allowed for car credit facilities"  cosine 0.8247
# The opposite question scores *higher* than a genuine rephrasing, and both
# retrieve the same chunk, so neither a similarity threshold nor the retrieval
# agreement check below can separate them. In a compliance system, serving the
# maximum when someone asked for the minimum is a real harm, so the safe default
# is to generate. Turn this on only for a low-stakes deployment.
CACHE_SEMANTIC_ENABLED = _bool("CACHE_SEMANTIC_ENABLED", False)
CACHE_SIM_THRESHOLD    = _float("CACHE_SIM_THRESHOLD", 0.96)
# A semantic hit must also agree with the fresh retrieval (Jaccard over chunk
# ids), and must not flip any polarity term. Both are necessary, neither is
# sufficient -- see the note above.
CACHE_MIN_DOC_OVERLAP  = _float("CACHE_MIN_DOC_OVERLAP", 0.67)

# ── Server ────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _int("API_PORT", 8001)
# Uvicorn worker processes. Each holds its own index copy in RAM, so scale by
# available memory; 2-4 is plenty to keep Python off the critical path.
WEB_WORKERS = _int("WEB_WORKERS", 1)

# Admission control. vLLM's continuous batching is the real queue, so this is a
# backpressure guard, not a throttle -- set it well above expected concurrency.
MAX_INFLIGHT   = _int("MAX_INFLIGHT",   256)
QUEUE_TIMEOUT_S = _float("QUEUE_TIMEOUT_S", 20.0)

# Query-embedding micro-batching: collect arrivals for this long, then run one
# ONNX forward for the whole group instead of 50 separate ones.
EMBED_BATCH_WINDOW_MS = _float("EMBED_BATCH_WINDOW_MS", 6.0)
EMBED_BATCH_MAX       = _int("EMBED_BATCH_MAX", 64)
EMBED_THREADS         = _int("EMBED_THREADS", 4)
# Ingestion embeds tens of thousands of chunks in one pass and wants every
# core. 0 lets ONNX Runtime size the pool from the machine.
EMBED_THREADS_INGEST  = _int("EMBED_THREADS_INGEST", 0)   # 0 = let ONNX decide
# Opt-in multiprocessing for the ingest embedding pass. 0 = single process
# (safe everywhere, ~30 chunks/s). 4 is roughly 4x faster on this machine.
EMBED_PARALLEL        = _int("EMBED_PARALLEL", 0)
