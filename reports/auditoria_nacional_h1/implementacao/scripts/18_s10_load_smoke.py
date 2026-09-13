"""Run an in-process concurrent load smoke against the real S10 release."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.api import APISettings, create_app
from vs_epl_krls.product import S10ProductService


async def _benchmark(args: argparse.Namespace) -> dict[str, object]:
    service = S10ProductService(
        args.artifact,
        args.manifest,
        selection_manifest=args.selection_manifest,
        procurement_report=args.procurement_report,
    )
    app = create_app(
        service,
        settings=APISettings(
            environment="test",
            api_key="load-smoke",
            rate_limit_per_minute=min(10_000, args.requests + args.warmup + 10),
        ),
    )
    transport = httpx.ASGITransport(app=app)
    latencies_ms: list[float] = []
    statuses: list[int] = []
    contract_failures = 0
    semaphore = asyncio.Semaphore(args.concurrency)
    headers = {"X-API-Key": "load-smoke"}
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        timeout=10,
    ) as client:
        for _ in range(args.warmup):
            response = await client.get("/v1/forecast", headers=headers)
            if response.status_code != 200:
                raise RuntimeError("load-smoke warmup failed")

        async def request_one() -> None:
            nonlocal contract_failures
            async with semaphore:
                started = time.perf_counter_ns()
                response = await client.get("/v1/forecast", headers=headers)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                latencies_ms.append(elapsed)
                statuses.append(response.status_code)
                if response.status_code == 200:
                    body = response.json()
                    point = body.get("forecast", {}).get("point")
                    if body.get("contract_version") != "1.0" or not isinstance(point, float):
                        contract_failures += 1

        started = time.perf_counter()
        await asyncio.gather(*(request_one() for _ in range(args.requests)))
        elapsed_seconds = time.perf_counter() - started

    values = np.asarray(latencies_ms, dtype=float)
    successful = sum(code == 200 for code in statuses)
    report: dict[str, object] = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successful": successful,
        "errors": args.requests - successful,
        "contract_failures": contract_failures,
        "elapsed_seconds": elapsed_seconds,
        "throughput_requests_per_second": args.requests / elapsed_seconds,
        "latency_ms": {
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        },
    }
    latency = report["latency_ms"]
    assert isinstance(latency, dict)
    report["gates"] = {
        "zero_http_errors": successful == args.requests,
        "zero_contract_failures": contract_failures == 0,
        "p95_below_200ms": float(latency["p95"]) < 200,
        "p99_below_300ms": float(latency["p99"]) < 300,
        "throughput_above_100rps": args.requests / elapsed_seconds > 100,
    }
    report["passed"] = all(report["gates"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--artifact", type=Path,
        default=ROOT / "artifacts" / "releases" / "s10_production_2026-08-16.joblib",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "releases" / "2026-08-16.json",
    )
    parser.add_argument(
        "--selection-manifest", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_selection" / "selection_manifest_h1.json",
    )
    parser.add_argument(
        "--procurement-report", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "procurement_backtest.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "load_smoke.json",
    )
    args = parser.parse_args()
    if not 1 <= args.requests <= 10_000 or not 1 <= args.concurrency <= 100:
        raise ValueError("requests and concurrency are outside safe smoke-test limits")
    payload = asyncio.run(_benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
