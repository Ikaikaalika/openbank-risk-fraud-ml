from __future__ import annotations

import asyncio
import time
import statistics
import httpx


async def bench(url: str, payload: dict, concurrency: int = 20, total: int = 200):
    latencies = []
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient):
        async with sem:
            t0 = time.perf_counter()
            r = await client.post(url, json=payload)
            r.raise_for_status()
            latencies.append(time.perf_counter() - t0)

    async with httpx.AsyncClient(timeout=5.0) as client:
        tasks = [asyncio.create_task(one(client)) for _ in range(total)]
        await asyncio.gather(*tasks)
    p95 = statistics.quantiles(latencies, n=100)[94]
    print(f"Requests: {total}, Concurrency: {concurrency}, p95: {p95*1000:.1f} ms, avg: {sum(latencies)/len(latencies)*1000:.1f} ms")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8080/score/credit")
    ap.add_argument("--type", choices=["credit","fraud"], default="credit")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--total", type=int, default=200)
    args = ap.parse_args()
    payload = {"loan_amnt": 10000, "int_rate": 13.5, "dti": 18.2, "term": 36} if args.type=="credit" else {"time": 2, "amount": 999.0, "amt_z": 0.9}
    asyncio.run(bench(args.endpoint, payload, args.concurrency, args.total))

