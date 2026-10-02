#!/usr/bin/env python3
"""
Load test. Answers the only question that matters: at N concurrent users, what
is the p50/p95 time to first token and time to complete answer?

TTFT is the number to watch. With streaming, that is what a user experiences as
"the system responded"; total time is how long the full cited answer takes to
finish rendering.

    python benchmark.py --users 50 --requests 200
    python benchmark.py --users 50 --mode concise --no-cache
    python benchmark.py --users 10 --endpoint http://server:8001
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

QUESTIONS = [
    "What is the loan to value ratio for credit facilities granted for motor vehicles?",
    "What are the minimum capital requirements for licensed commercial banks under Basel III?",
    "What must a bank report following a cyber security event?",
    "What is the liquidity coverage ratio requirement?",
    "What are the requirements for outsourcing business operations?",
    "How is the risk weighted amount for operational risk computed?",
    "What are the fitness and propriety requirements for directors?",
    "What is the definition of liquid assets under the Banking Act?",
    "What are the rules on foreign currency borrowings by licensed banks?",
    "What are the requirements for the appointment of agents by a bank?",
    "What does the deposit insurance scheme cover?",
    "What are the stress testing guidelines for licensed banks?",
    "What are the maximum amount of accommodation limits?",
    "What are the requirements on valuation of immovable property?",
    "What is required for the implementation of SLFRS 9?",
    "What are the annual licence fee requirements for commercial banks?",
]


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


async def one_request(client: httpx.AsyncClient, url: str, question: str,
                      mode: str) -> dict:
    """One streamed request. Records when the first token actually arrives."""
    t0, ttft, tokens = time.perf_counter(), None, 0
    body = {"question": question, "mode": mode}
    cache_state = "miss"
    try:
        async with client.stream("POST", f"{url}/ask/stream", json=body) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return {"ok": False, "status": resp.status_code}
            event = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event == "token":
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    tokens += 1
                elif line.startswith("data:") and event == "done":
                    try:
                        cache_state = json.loads(line[5:]).get("cache", "miss")
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:                                        # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "ttft_ms" : ttft if ttft is not None else (time.perf_counter() - t0) * 1000,
        "total_ms": (time.perf_counter() - t0) * 1000,
        "chunks"  : tokens,
        "cache"   : cache_state,
    }


async def run(url: str, users: int, total: int, mode: str, unique: bool):
    limits = httpx.Limits(max_connections=users + 16,
                          max_keepalive_connections=users + 16)
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0),
                                 limits=limits) as client:
        try:
            r = await client.get(f"{url}/health")
            r.raise_for_status()
            print(f"target  : {url}  ({r.json()['chunks_indexed']} chunks indexed)")
        except Exception as exc:                                    # noqa: BLE001
            raise SystemExit(f"cannot reach {url}: {exc}")

        sem     = asyncio.Semaphore(users)
        results = []

        async def worker(i: int):
            q = random.choice(QUESTIONS)
            if unique:
                # Defeat the answer cache so this measures cold generation only.
                q = f"{q} (case reference {i})"
            async with sem:
                results.append(await one_request(client, url, q, mode))

        print(f"users   : {users} concurrent")
        print(f"requests: {total}   mode: {mode}   cache-defeating: {unique}\n")

        t0 = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(total)))
        wall = time.perf_counter() - t0

    ok      = [r for r in results if r.get("ok")]
    bad     = [r for r in results if not r.get("ok")]
    ttfts   = [r["ttft_ms"] for r in ok]
    totals  = [r["total_ms"] for r in ok]
    hits    = sum(1 for r in ok if r.get("cache") in ("exact", "semantic"))

    print("=" * 58)
    print(f"completed        : {len(ok)}/{total}   failed: {len(bad)}")
    print(f"wall clock       : {wall:.1f}s   throughput: {len(ok)/wall:.1f} req/s")
    print(f"cache hits       : {hits}/{len(ok)}")
    print("-" * 58)
    print(f"{'':<10}{'p50':>10}{'p95':>10}{'p99':>10}{'mean':>10}")
    for name, vals in (("TTFT ms", ttfts), ("total ms", totals)):
        if vals:
            print(f"{name:<10}{pct(vals,.50):>10.0f}{pct(vals,.95):>10.0f}"
                  f"{pct(vals,.99):>10.0f}{statistics.mean(vals):>10.0f}")
    print("=" * 58)

    if bad:
        kinds: dict[str, int] = {}
        for r in bad:
            key = str(r.get("status") or r.get("error"))[:60]
            kinds[key] = kinds.get(key, 0) + 1
        print("failures:")
        for k, v in kinds.items():
            print(f"  {v:4d}  {k}")

    if ttfts:
        print(f"\nSLO: p95 TTFT {pct(ttfts,.95):.0f} ms "
              f"{'PASS' if pct(ttfts,.95) < 1000 else 'FAIL'} (<1000 ms)")
        print(f"     p95 total {pct(totals,.95):.0f} ms "
              f"{'PASS' if pct(totals,.95) < 1000 else 'over 1 s'} (<1000 ms)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://localhost:8001")
    ap.add_argument("--users", type=int, default=50, help="Concurrent in-flight requests.")
    ap.add_argument("--requests", type=int, default=200, help="Total requests to send.")
    ap.add_argument("--mode", default="balanced", choices=["concise", "balanced", "detailed"])
    ap.add_argument("--no-cache", action="store_true",
                    help="Make every question unique so nothing hits the answer cache.")
    args = ap.parse_args()

    asyncio.run(run(args.endpoint.rstrip("/"), args.users, args.requests,
                    args.mode, args.no_cache))


if __name__ == "__main__":
    main()
