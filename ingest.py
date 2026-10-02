#!/usr/bin/env python3
"""
Ingestion: PDFs -> chunks -> embeddings -> vector index.

Differences from the prototype's ingest that matter for serve-time latency:

* Pages are split into overlapping chunks instead of being stored whole. A hit
  on a 6000-character page used to drag the whole page into the prompt; now it
  drags ~1400 characters. Prefill cost is linear in that, so this is the largest
  TTFT lever available at ingest time.

* Vision transcription runs with real concurrency. The prototype held a
  Semaphore(1) around the vLLM vision call, so every scanned page was
  transcribed strictly one at a time.

* Page images are rendered in memory for the vision pass and thrown away. They
  are not needed at serve time -- the transcribed *text* is what gets indexed,
  so the serving path stays text-only and vLLM can drop the vision tower
  entirely. Pass --save-images to keep them on disk anyway.

* Output is a plain vectors.npy plus meta.jsonl, not a ChromaDB directory. An
  approximate HNSW index is only built above ~150k chunks, where exact search
  stops being the fastest option.

Run it through the ingest overlay, which switches vLLM's vision model on:

    docker compose -f docker-compose.yml -f docker-compose.ingest.yml \
      run --rm app python ingest.py

    python ingest.py                # full run: extract text, transcribe scans
    python ingest.py --no-vision    # skip scanned pages, minutes instead of hours
    python ingest.py --save-images  # also keep the rendered page JPEGs on disk

There is also an escape hatch for rebuilding without paying for OCR twice. The
prototype's ChromaDB already holds transcribed text for every scanned page, so
if you later need to re-chunk or change the embedding model, mount that volume
and read from it instead of re-running the vision model:

    # in docker-compose.yml, add to the app service:
    #   volumes:
    #     - prototype_chroma:/chroma:ro
    # and at the top level:
    #   volumes:
    #     prototype_chroma:
    #       external: true
    #       name: compliance_rag_prototype_chroma_data
    python ingest.py --from-chroma      # defaults to /chroma

That path reads chroma.sqlite3 with the standard library, so it needs no
chromadb package, and it writes the same index a fresh run would.
"""
from __future__ import annotations

import os

# Must run before onnxruntime is imported, which happens via fastembed further
# down.
#
# The serving image pins OMP_NUM_THREADS=1 because the API gets its parallelism
# from concurrent requests. Ingestion is the opposite case -- one batch pass over
# every chunk in the corpus -- so it wants more than one thread. But not one per
# core: measured on this box, 28 OpenMP threads on bge-small's small forward pass
# thrashed at 2.1 chunks/s, roughly 15x SLOWER than leaving it alone. Eight is
# the sweet spot. Set here rather than only in compose so that running ingest
# through any service still gets it.
if os.environ.get("OMP_NUM_THREADS", "1") == "1":
    os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz          # PyMuPDF
import httpx
import numpy as np
from tqdm import tqdm

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("ingest.log", encoding="utf-8")],
)
log = logging.getLogger("ingest")

VISION_PROMPT = (
    "This is a page from a compliance or regulatory document. Transcribe ALL "
    "text visible on this page exactly as written, including every heading, "
    "paragraph, bullet, table cell and footnote. Do not summarise."
)


# -- helpers ------------------------------------------------------------------
def file_hash(path: Path) -> str:
    st = path.stat()
    h = hashlib.md5(f"{st.st_size}:{int(st.st_mtime)}".encode())
    with path.open("rb") as fh:
        h.update(fh.read(65536))
    return h.hexdigest()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Split on paragraph boundaries where possible so a chunk rarely cuts a
    sentence in half -- the model has to quote these verbatim, and a truncated
    quote is a broken citation.
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", "\n", ". "):
                cut = window.rfind(sep)
                if cut > size * 0.5:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_progress() -> dict:
    if cfg.PROGRESS_FILE.exists():
        try:
            return json.loads(cfg.PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            return {}
    return {}


def save_progress(progress: dict):
    cfg.PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cfg.PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


# -- phase 1: text extraction -------------------------------------------------
def extract_pdf(pdf_path: Path, save_images: bool) -> tuple[list[dict], list[dict]]:
    """
    Returns (pages_with_text, pages_needing_vision). A page needing vision
    carries its rendered JPEG bytes so phase 2 does not reopen the PDF.
    """
    prefix = hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()[:8]
    try:
        folder = str(pdf_path.parent.relative_to(cfg.DATASET_DIR.resolve()))
    except ValueError:
        folder = pdf_path.parent.name

    good, needs_vision = [], []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:                                       # noqa: BLE001
        log.error("  x  %s -- %s", pdf_path.name, exc)
        return [], []

    total = len(doc)
    try:
        for i in range(total):
            page = doc[i]
            try:
                text = page.get_text("text").strip()
            except Exception:                                      # noqa: BLE001
                text = ""

            rec = {
                "doc_id"     : prefix,
                "source"     : str(pdf_path.resolve()),
                "filename"   : pdf_path.name,
                "folder"     : folder,
                "page"       : i + 1,
                "total_pages": total,
                "text"       : text,
            }

            if len(text) >= cfg.MIN_TEXT_LEN:
                good.append(rec)
                continue

            mat = fitz.Matrix(cfg.IMAGE_DPI / 72, cfg.IMAGE_DPI / 72)
            jpeg = page.get_pixmap(matrix=mat, alpha=False).tobytes("jpeg", jpg_quality=80)
            if save_images:
                cfg.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                (cfg.IMAGES_DIR / f"{prefix}_p{i + 1:04d}.jpg").write_bytes(jpeg)
            needs_vision.append({**rec, "_jpeg": jpeg})
    finally:
        doc.close()

    return good, needs_vision


# -- phase 2: vision transcription --------------------------------------------
async def transcribe_all(pending: list[dict]) -> list[dict]:
    """Transcribe scanned pages against vLLM, VISION_CONCURRENCY at a time."""
    if not pending:
        return []

    sem     = asyncio.Semaphore(cfg.VISION_CONCURRENCY)
    headers = {"Authorization": f"Bearer {cfg.VLLM_API_KEY}"} if cfg.VLLM_API_KEY else {}
    host    = cfg.VLLM_HOSTS[0]
    done: list[dict] = []
    bar = tqdm(total=len(pending), desc="Vision", unit="page", ncols=80)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(180.0, connect=10.0),
        limits=httpx.Limits(max_connections=cfg.VISION_CONCURRENCY * 2),
        headers=headers,
    ) as client:

        async def one(rec: dict):
            b64 = base64.b64encode(rec.pop("_jpeg")).decode()
            async with sem:
                try:
                    r = await client.post(f"{host}/v1/chat/completions", json={
                        "model": cfg.LLM_MODEL,
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ]}],
                        "temperature": 0.0,
                        "max_tokens": 1536,
                    })
                    r.raise_for_status()
                    out = r.json()["choices"][0]["message"]["content"].strip()
                    if out:
                        rec["text"] = out
                except Exception as exc:                           # noqa: BLE001
                    log.warning("vision failed %s p%d: %s", rec["filename"], rec["page"], exc)
                finally:
                    bar.update(1)
            if rec["text"].strip():
                done.append(rec)

        await asyncio.gather(*(one(r) for r in pending))

    bar.close()
    return done


# -- phase 3+4: chunk, embed, build index -------------------------------------
HNSW_THRESHOLD = 150_000   # keep in step with Retriever.HNSW_THRESHOLD


def build_index(pages: list[dict]):
    from fastembed import TextEmbedding

    records = []
    for p in pages:
        for j, piece in enumerate(chunk_text(p["text"], cfg.CHUNK_CHARS, cfg.CHUNK_OVERLAP)):
            records.append({
                "id"         : f"{p['doc_id']}_p{p['page']:04d}_c{j:02d}",
                "source"     : p["source"],
                "filename"   : p["filename"],
                "folder"     : p["folder"],
                "page"       : p["page"],
                "total_pages": p["total_pages"],
                "text"       : piece,
            })

    # Stable ordering keeps chunk ids -- and therefore the canonical prompt
    # ordering the prefix cache relies on -- reproducible across rebuilds.
    records.sort(key=lambda r: r["id"])
    if not records:
        log.warning("Nothing to index.")
        return

    log.info("Embedding %d chunks with %s ...", len(records), cfg.EMBED_MODEL)
    # Ingestion is a batch job, so give ONNX every core. The serving path does
    # the opposite (threads=1) because there parallelism comes from concurrent
    # requests and intra-op threads would only contend with each other.
    # Offline loading is enforced by HF_HUB_OFFLINE, set in config.py.
    model = TextEmbedding(model_name=cfg.EMBED_MODEL, cache_dir=cfg.EMBED_CACHE,
                          threads=cfg.EMBED_THREADS_INGEST or None)

    # parallel=None keeps everything in this process (safe everywhere).
    # parallel=N forks N workers, which is roughly N times faster on a big
    # corpus but relies on multiprocessing behaving inside the container --
    # hence opt-in via EMBED_PARALLEL rather than on by default.
    embed_kw = {"batch_size": cfg.EMBED_BATCH_SIZE}
    if cfg.EMBED_PARALLEL > 0:
        embed_kw["parallel"] = cfg.EMBED_PARALLEL
        log.info("Embedding with %d parallel workers", cfg.EMBED_PARALLEL)

    # Consume the generator one vector at a time and tick the bar per chunk.
    # Batching the calls and updating once per batch made the bar sit at 0% for
    # minutes before jumping, which is indistinguishable from a hang -- and this
    # stage runs right after a 20 minute OCR pass, so "is it stuck?" is exactly
    # the question the operator is asking. tqdm throttles its own redraws, so
    # per-item updates cost nothing.
    texts = [r["text"] for r in records]
    vecs_out = []
    bar = tqdm(total=len(records), desc="Embed", unit="chunk", ncols=80,
               smoothing=0.05, mininterval=0.5)
    for vec in model.embed(texts, **embed_kw):
        vecs_out.append(vec)
        bar.update(1)
    bar.close()

    if len(vecs_out) != len(records):
        log.error("Embedding returned %d vectors for %d chunks", len(vecs_out), len(records))
        sys.exit(1)

    vectors = np.asarray(vecs_out, dtype=np.float32)
    dim     = int(vectors.shape[1])
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)

    cfg.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cfg.VECTORS_PATH, vectors)

    # Exact search over this many vectors is about a millisecond and has perfect
    # recall, so an approximate index is only worth building for a much larger
    # corpus. hnswlib stays an optional dependency because of that.
    backend = "exact"
    if len(records) >= HNSW_THRESHOLD:
        try:
            import hnswlib
            log.info("Corpus is large (%d chunks) -- building HNSW index "
                     "(M=%d, ef_construction=%d) ...",
                     len(records), cfg.HNSW_M, cfg.HNSW_EF_CONSTRUCTION)
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=len(records),
                             ef_construction=cfg.HNSW_EF_CONSTRUCTION,
                             M=cfg.HNSW_M)
            index.add_items(vectors, np.arange(len(records)),
                            num_threads=cfg.INGEST_WORKERS)
            index.save_index(str(cfg.INDEX_PATH))
            backend = "hnsw"
        except ImportError:
            log.warning("hnswlib not installed -- staying on exact search. "
                        "`pip install hnswlib` if query latency becomes a problem.")
    else:
        cfg.INDEX_PATH.unlink(missing_ok=True)

    with cfg.META_PATH.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cfg.MANIFEST_PATH.write_text(json.dumps({
        "chunks"      : len(records),
        "pages"       : len(pages),
        "dim"         : dim,
        "backend"     : backend,
        "embed_model" : cfg.EMBED_MODEL,
        "chunk_chars" : cfg.CHUNK_CHARS,
        "overlap"     : cfg.CHUNK_OVERLAP,
        "built_at"    : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")

    log.info("Index written: %d chunks from %d pages, backend=%s -> %s",
             len(records), len(pages), backend, cfg.INDEX_DIR)


# -- page checkpoint ----------------------------------------------------------
def save_pages(pages: list[dict], path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.replace(path)            # atomic, so a crash never leaves a half file
    log.info("Checkpoint written: %d pages -> %s", len(pages), path)


def load_pages(path) -> list[dict]:
    if not path.exists():
        log.error("No checkpoint at %s. Run ingest.py without --from-pages first.", path)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as fh:
        pages = [json.loads(line) for line in fh]
    log.info("Loaded %d pages from checkpoint %s", len(pages), path)
    return pages


# -- corpus sources -----------------------------------------------------------
CHROMA_COLLECTION = "compliance_mm"


def pages_from_chroma(chroma_dir: Path) -> list[dict]:
    """
    Reuse the prototype's finished corpus.

    Vision transcription of ~1000 scanned PDFs is by far the slowest part of a
    cold ingest and the prototype already paid for it, so this lifts the text
    straight out rather than re-running the vision model.

    Read with the standard library rather than the chromadb package. Chroma
    pulls in onnxruntime, grpcio, opentelemetry and more -- a lot of image
    weight for one migration that touches four columns. Its on-disk format here
    is an ordinary SQLite file: `embeddings` holds one row per stored page and
    `embedding_metadata` holds that page's fields as key/value rows, with the
    page text under the key `chroma:document`.
    """
    db = chroma_dir / "chroma.sqlite3" if chroma_dir.is_dir() else chroma_dir
    if not db.exists():
        log.error("No chroma.sqlite3 at %s", db)
        sys.exit(1)

    log.info("Reading ChromaDB at %s", db)
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """
            SELECT e.embedding_id, m.key, m.string_value, m.int_value
            FROM embeddings e
            JOIN embedding_metadata m ON m.id = e.id
            JOIN segments    s ON s.id = e.segment_id
            JOIN collections c ON c.id = s.collection
            WHERE c.name = ?
            """,
            (CHROMA_COLLECTION,),
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        log.error("Could not read %s: %s", db, exc)
        sys.exit(1)
    finally:
        con.close()

    if not rows:
        log.error("Collection %r is empty or missing in %s", CHROMA_COLLECTION, db)
        sys.exit(1)

    grouped: dict[str, dict] = {}
    for emb_id, key, sval, ival in rows:
        grouped.setdefault(emb_id, {})[key] = sval if sval is not None else ival

    pages, skipped = [], 0
    for meta in tqdm(grouped.values(), desc="Chroma", unit="rec", ncols=80):
        text = (meta.get("chroma:document") or "").strip()
        if not text:
            skipped += 1
            continue
        source = meta.get("source") or ""
        pages.append({
            # Keyed on the source path so chunk ids stay stable across rebuilds,
            # which is what keeps the prompt prefix cacheable.
            "doc_id"     : hashlib.md5(source.encode()).hexdigest()[:8],
            "source"     : source,
            "filename"   : meta.get("filename") or "unknown.pdf",
            "folder"     : meta.get("folder") or "",
            "page"       : int(meta.get("page") or 1),
            "total_pages": int(meta.get("total_pages") or 1),
            "text"       : text,
        })

    log.info("Imported %d pages (%d had no text and were skipped)", len(pages), skipped)
    return pages


def pages_from_pdfs(use_vision: bool, workers: int, save_images: bool) -> list[dict]:
    pdfs = sorted({p.resolve() for pat in ("*.pdf", "*.PDF")
                   for p in cfg.DATASET_DIR.rglob(pat)})
    if not pdfs:
        log.error("No PDFs under %s", cfg.DATASET_DIR.resolve())
        sys.exit(1)

    # HNSW is rebuilt whole every run, so every PDF is re-read from disk (cheap).
    # The progress file exists so a *resumed* run can skip re-transcribing pages
    # the vision model already handled -- that is the expensive part.
    log.info("%d PDFs found", len(pdfs))

    good, needs_vision = [], []
    bar = tqdm(total=len(pdfs), desc="Extract", unit="pdf", ncols=80)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pdf") as exe:
        futures = {exe.submit(extract_pdf, p, save_images): p for p in pdfs}
        for fut in as_completed(futures):
            try:
                g, nv = fut.result()
                good.extend(g)
                needs_vision.extend(nv)
            except Exception as exc:                               # noqa: BLE001
                log.error("extract failed for %s: %s", futures[fut].name, exc)
            bar.update(1)
            bar.set_postfix(pages=len(good), ocr=len(needs_vision), refresh=False)
    bar.close()

    if needs_vision and use_vision:
        log.info("%d pages have little or no extractable text -- transcribing with "
                 "the vision model (%d concurrent)", len(needs_vision), cfg.VISION_CONCURRENCY)
        good.extend(asyncio.run(transcribe_all(needs_vision)))
    elif needs_vision:
        log.warning("%d pages skipped (--no-vision)", len(needs_vision))

    save_progress({str(p): {"hash": file_hash(p)} for p in pdfs})
    return good


# -- entry point --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="Ignore the progress file and rebuild everything.")
    ap.add_argument("--no-vision", action="store_true",
                    help="Skip vision transcription of scanned pages.")
    ap.add_argument("--workers", type=int, default=cfg.INGEST_WORKERS)
    ap.add_argument("--save-images", action="store_true",
                    help="Keep rendered page JPEGs on disk.")
    # nargs="?"/const so `--from-chroma` works with no argument. Git Bash on
    # Windows rewrites a bare "/chroma" into a host path (C:/.../chroma), which
    # is confusing to debug; giving the flag a default removes the trap.
    ap.add_argument("--from-pages", action="store_true",
                    help="Skip extraction and vision entirely; rebuild the index "
                         "from the pages.jsonl checkpoint of a previous run. Use "
                         "this to retry a failed embedding step without paying "
                         "for OCR again.")
    ap.add_argument("--from-chroma", nargs="?", const="/chroma", metavar="DIR",
                    help="Import an existing ChromaDB corpus instead of re-reading "
                         "PDFs. Defaults to /chroma, where the compose file mounts "
                         "the prototype's volume.")
    args = ap.parse_args()

    t0 = time.time()
    log.info("=" * 64)
    log.info("  VLLMRAG -- ingestion")
    log.info("=" * 64)

    if args.force:
        cfg.PROGRESS_FILE.unlink(missing_ok=True)

    if args.from_pages:
        pages = load_pages(cfg.PAGES_PATH)
    elif args.from_chroma:
        pages = pages_from_chroma(Path(args.from_chroma))
    else:
        pages = pages_from_pdfs(not args.no_vision, args.workers, args.save_images)

    log.info("Collected %d pages with text", len(pages))

    # Checkpoint before embedding. Vision transcription is the expensive stage --
    # tens of minutes of GPU time -- and embedding is the stage most likely to
    # fail on a fresh machine. Writing the pages out first means a failure there
    # costs a retry of the cheap half only: rerun with --from-pages.
    if not args.from_pages:
        save_pages(pages, cfg.PAGES_PATH)

    build_index(pages)
    log.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
