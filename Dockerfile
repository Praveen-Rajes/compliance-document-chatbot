# Air-gapped build. Same pattern as the prototype: nothing here reaches the
# network. Wheels come from ./packages (populated by `pip download` on a machine
# that does have internet) and the embedding model comes from ./embedding_cache.
#
# Note there is no `apt-get` and no HEALTHCHECK using curl -- both would need
# network access at build time. The healthcheck lives in docker-compose.yml and
# uses the Python already in the image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Serving default. fastembed runs ONNX on CPU; left unbounded it spawns a
    # thread per core and they contend, because the API already gets its
    # parallelism from concurrent requests. ingest.py raises this to 8 for its
    # batch pass -- see the comment at the top of that file.
    OMP_NUM_THREADS=1 \
    # Must match where the embedding cache is copied below.
    EMBED_CACHE_DIR=/app/embedding_cache \
    # Air-gap enforcement. Without these, fastembed compares the cached model
    # against HuggingFace metadata, decides the file sizes do not match, and
    # blocks trying to re-download -- which on an isolated network hangs with no
    # error at all. config.py sets the same variables for bare-metal runs.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY requirements.txt .
COPY packages/ /packages/
RUN pip install --no-cache-dir --no-index --find-links=/packages -r requirements.txt

COPY embedding_cache/ /app/embedding_cache/

COPY . .

EXPOSE 8001

CMD ["python", "server.py"]
