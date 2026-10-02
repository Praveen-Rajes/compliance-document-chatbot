"""
Async vLLM client.

Everything here is non-blocking. The prototype used `requests` inside a
4-thread pool, which capped real concurrency at 4 no matter what the semaphore
said; a single httpx.AsyncClient with a large pool lets all in-flight requests
sit on the socket at once and lets vLLM's continuous batcher decide the order.

Multiple backends are supported so the 2x L40S box can run one replica per GPU
(data parallel) instead of tensor-parallel across both. For a ~5B model, TP=2
spends more on all-reduce than it wins back, and two independent replicas keep
serving if one GPU is pulled for another workload.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from typing import AsyncIterator

import httpx

import config as cfg

log = logging.getLogger(__name__)


class VLLMClient:
    def __init__(self, hosts: list[str] | None = None):
        self.hosts    = hosts or cfg.VLLM_HOSTS
        self._inflight = {h: 0 for h in self.hosts}
        self._rr       = itertools.cycle(self.hosts)
        self._client: httpx.AsyncClient | None = None
        self.requests  = 0
        self.failures  = 0

    async def start(self):
        limits = httpx.Limits(
            max_connections=cfg.MAX_INFLIGHT + 32,
            max_keepalive_connections=cfg.MAX_INFLIGHT + 32,
            keepalive_expiry=60.0,
        )
        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(cfg.VLLM_TIMEOUT, connect=cfg.VLLM_CONNECT_TIMEOUT),
            headers=({"Authorization": f"Bearer {cfg.VLLM_API_KEY}"} if cfg.VLLM_API_KEY else {}),
        )

    async def close(self):
        if self._client:
            await self._client.aclose()

    # ── backend selection ─────────────────────────────────────────────────────
    def _pick(self) -> str:
        """Least in-flight, ties broken round-robin."""
        if len(self.hosts) == 1:
            return self.hosts[0]
        low = min(self._inflight.values())
        for _ in range(len(self.hosts)):
            h = next(self._rr)
            if self._inflight[h] == low:
                return h
        return self.hosts[0]

    async def wait_ready(self, max_wait_s: int = 600, interval_s: int = 5):
        """
        vLLM needs a minute or more to load weights and capture CUDA graphs.
        Every backend must answer before the app reports itself ready.
        """
        deadline = time.time() + max_wait_s
        pending  = list(self.hosts)
        while pending and time.time() < deadline:
            still = []
            for host in pending:
                try:
                    r = await self._client.get(f"{host}/v1/models", timeout=10.0)
                    r.raise_for_status()
                    served = [m["id"] for m in r.json().get("data", [])]
                    log.info("vLLM ready at %s — serving %s", host, served)
                except Exception as exc:                          # noqa: BLE001
                    log.warning("vLLM not ready at %s (%s)", host, exc)
                    still.append(host)
            pending = still
            if pending:
                await asyncio.sleep(interval_s)
        if pending:
            raise RuntimeError(f"vLLM backends never became ready: {pending}")

    # ── generation ────────────────────────────────────────────────────────────
    def _payload(self, messages: list[dict], max_tokens: int, stream: bool) -> dict:
        return {
            "model"      : cfg.LLM_MODEL,
            "messages"   : messages,
            "temperature": cfg.TEMPERATURE,
            "max_tokens" : max_tokens,
            "stream"     : stream,
            # Ask vLLM for cache diagnostics on the non-streaming path so /stats
            # can show the real prefix-cache hit rate rather than a guess.
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }

    async def complete(self, messages: list[dict], max_tokens: int) -> tuple[str, dict, str | None]:
        host = self._pick()
        self._inflight[host] += 1
        self.requests += 1
        try:
            r = await self._client.post(
                f"{host}/v1/chat/completions",
                json=self._payload(messages, max_tokens, stream=False),
            )
            r.raise_for_status()
            body = r.json()
            choice = body["choices"][0]
            return (choice["message"]["content"], body.get("usage", {}),
                    choice.get("finish_reason"))
        except Exception:
            self.failures += 1
            raise
        finally:
            self._inflight[host] -= 1

    async def stream(self, messages: list[dict], max_tokens: int
                     ) -> AsyncIterator[tuple[str, dict | None, str | None]]:
        """
        Yields (delta_text, usage, finish_reason).

        finish_reason arrives on the last chunk of a choice. "length" means the
        model was cut off at max_tokens rather than finishing its answer, which
        the caller needs to know so it can say so instead of presenting a
        half-written answer as complete.
        """
        host = self._pick()
        self._inflight[host] += 1
        self.requests += 1
        try:
            async with self._client.stream(
                "POST",
                f"{host}/v1/chat/completions",
                json=self._payload(messages, max_tokens, stream=True),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage")
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            yield delta, None, None
                        if choice.get("finish_reason"):
                            yield "", None, choice["finish_reason"]
                    if usage:
                        yield "", usage, None
        except Exception:
            self.failures += 1
            raise
        finally:
            self._inflight[host] -= 1

    def stats(self) -> dict:
        return {
            "backends" : self.hosts,
            "inflight" : dict(self._inflight),
            "requests" : self.requests,
            "failures" : self.failures,
        }


llm = VLLMClient()
