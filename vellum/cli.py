"""Command-line entry points.

    vellum sweep   --label fp16 --concurrency 1 2 4 8 16 32 64
    vellum offline --backend hf --model Qwen/Qwen2.5-3B-Instruct --label hf-fp16
    vellum plot    results/
    vellum report  results/
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import aiohttp

from .client import results_to_rows, run_level, summarize


async def _detect_model(base_url: str) -> str:
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{base_url}/v1/models") as r:
            r.raise_for_status()
            return (await r.json())["data"][0]["id"]


async def _sweep(args: argparse.Namespace) -> None:
    base = args.base_url.rstrip("/")
    model = args.model or await _detect_model(base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.label}] model={model}  in={args.input_len}  out={args.output_len}")
    # Warmup: CUDA graph capture, allocator growth, etc. should not pollute level 1.
    await run_level(base, model, num_requests=args.warmup, input_len=args.input_len,
                    output_len=args.output_len, concurrency=max(1, args.warmup),
                    shared_prefix_len=args.shared_prefix_len, seed=args.seed + 1)

    levels = [("concurrency", c) for c in args.concurrency or []]
    levels += [("request_rate", r) for r in args.request_rate or []]
    summaries = []
    raw_path = out_dir / f"{args.label}_requests.csv"
    with raw_path.open("w", newline="") as fh:
        writer = None
        for i, (mode, value) in enumerate(levels):
            n = args.num_requests or max(args.min_requests, int(value * args.requests_per_slot))
            kw = {mode: value}
            results, samples, wall = await run_level(
                base, model, num_requests=n, input_len=args.input_len,
                output_len=args.output_len, shared_prefix_len=args.shared_prefix_len,
                # Fresh prompts per level: reusing them would hand later levels
                # full-prompt prefix-cache hits from earlier ones.
                seed=args.seed + 1000 * (i + 1), **kw,
            )
            s = summarize(results, samples, wall, label=args.label, model=model,
                          mode=mode, level=value, input_len=args.input_len,
                          output_len=args.output_len, shared_prefix_len=args.shared_prefix_len)
            summaries.append(s)
            print(
                f"  {mode}={value:<6} ok={s['num_ok']}/{s['num_requests']}  "
                f"out_tok/s={s['output_token_throughput']:8.1f}  "
                f"TTFT p50={_f(s['p50_ttft_ms'])} p99={_f(s['p99_ttft_ms'])}ms  "
                f"TPOT p50={_f(s['p50_tpot_ms'])}ms  "
                f"KV max={_f(s['max_kv_cache_usage'], pct=True)}"
            )
            for row in results_to_rows(results):
                row.update(mode=mode, level=value)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)

    meta = {"label": args.label, "experiment": args.experiment, "model": model, "base_url": base,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "server_args": args.server_args}
    (out_dir / f"{args.label}.json").write_text(json.dumps({"meta": meta, "levels": summaries}, indent=2))
    print(f"  -> {out_dir / (args.label + '.json')}")


def _f(v, pct: bool = False) -> str:
    if v is None:
        return "  n/a"
    return f"{v * 100:4.0f}%" if pct else f"{v:6.1f}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vellum")
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("sweep", help="benchmark a running server across load levels")
    sw.add_argument("--base-url", default="http://localhost:8000")
    sw.add_argument("--model", help="defaults to the first model the server reports")
    sw.add_argument("--label", required=True, help="name of this server config, e.g. fp16, awq")
    sw.add_argument("--experiment", default="main",
                    help="runs with the same experiment tag are compared head-to-head")
    sw.add_argument("--concurrency", type=int, nargs="*", help="closed-loop in-flight request counts")
    sw.add_argument("--request-rate", type=float, nargs="*", help="open-loop Poisson rates (req/s)")
    sw.add_argument("--input-len", type=int, default=512)
    sw.add_argument("--output-len", type=int, default=128)
    sw.add_argument("--shared-prefix-len", type=int, default=0,
                    help="tokens of common prefix across requests (prefix-caching experiment)")
    sw.add_argument("--num-requests", type=int, help="fixed count per level (default: scales with level)")
    sw.add_argument("--requests-per-slot", type=float, default=4)
    sw.add_argument("--min-requests", type=int, default=32)
    sw.add_argument("--warmup", type=int, default=4)
    sw.add_argument("--seed", type=int, default=0)
    sw.add_argument("--server-args", default="", help="recorded in metadata only")
    sw.add_argument("--out", default="results")

    from .offline import add_parser as add_offline_parser
    add_offline_parser(sub)

    pl = sub.add_parser("plot", help="render charts from sweep JSON files")
    pl.add_argument("results_dir")
    rp = sub.add_parser("report", help="print a markdown summary table")
    rp.add_argument("results_dir")
    rp.add_argument("--gpu-cost-per-hour", type=float,
                    help="GPU rental price in USD/hr, used for cost per 1M tokens")

    args = p.parse_args(argv)
    if args.cmd == "sweep":
        if not (args.concurrency or args.request_rate):
            sys.exit("sweep: pass --concurrency and/or --request-rate")
        asyncio.run(_sweep(args))
    elif args.cmd == "offline":
        from .offline import main as offline_main
        offline_main(args)
    elif args.cmd == "plot":
        from .plot import plot_dir
        for path in plot_dir(Path(args.results_dir)):
            print(path)
    elif args.cmd == "report":
        from .report import report_dir
        print(report_dir(Path(args.results_dir), args.gpu_cost_per_hour))


if __name__ == "__main__":
    main()
