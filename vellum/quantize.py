"""`vellum quantize`: AWQ-quantize the from-scratch Mistral, and measure what it costs in quality.

    vellum quantize --model mistralai/Mistral-7B-Instruct-v0.3 --out weights/awq \\
        --quality results/NVIDIA-L4/awq_quality.json

Calibrates on WikiText-2 train, writes a 4-bit checkpoint that `--backend scratch --rung awq`
loads, and with --quality reports WikiText-2 test perplexity for the bf16 model, AWQ and plain
round-to-nearest (RTN). The gap between the last two is what AWQ's scale search buys.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def add_parser(sub) -> None:
    p = sub.add_parser("quantize", help="AWQ-quantize the from-scratch Mistral to 4 bits")
    p.add_argument("--model", required=True, help="Hub repo id or folder with Mistral's consolidated checkpoint")
    p.add_argument("--out", required=True, help="folder for the quantized checkpoint")
    p.add_argument("--samples", type=int, default=32, help="calibration windows")
    p.add_argument("--length", type=int, default=512, help="tokens per calibration window")
    p.add_argument("--grid", type=int, default=20, help="alpha values AWQ searches per layer group")
    p.add_argument("--quality", help="also write WikiText-2 perplexity (bf16, AWQ, RTN) to this JSON file")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)


def main(args: argparse.Namespace) -> None:
    import torch

    from .mistral import awq_quantize, load_model, load_tokenizer, perplexity, save_quantized
    from .mistral.awq import calibration_batch, wikitext

    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.model)

    def tokens(split: str) -> torch.Tensor:
        return torch.tensor(tokenizer.encode(wikitext(split), bos=False, eos=False))

    calibration = calibration_batch(tokens("train"), args.samples, args.length, args.seed).to(device)
    test = tokens("test") if args.quality else None
    details = {"method": "awq", "bits": 4, "group_size": 128, "source": str(args.model),
               "calibration": f"wikitext-2 train, {args.samples} x {args.length} tokens", "grid": args.grid}

    model = load_model(args.model, device=device, dtype=torch.bfloat16)
    quality = {}
    if test is not None:
        quality["bf16"] = perplexity(model, test)
        print(f"  perplexity bf16: {quality['bf16']:.3f}", flush=True)
    t0 = time.perf_counter()
    awq_quantize(model, calibration, n_grid=args.grid, log=lambda msg: print(f"  awq {msg}", flush=True))
    details["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  -> {save_quantized(model, args.out, details)}  ({details['seconds']} s)", flush=True)
    if test is None:
        return

    quality["awq"] = perplexity(model, test)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rtn = load_model(args.model, device=device, dtype=torch.bfloat16)
    awq_quantize(rtn, calibration, search=False)
    quality["rtn"] = perplexity(rtn, test)
    report = {"model": str(args.model), "dataset": "wikitext-2-raw-v1 test", "window": 2048,
              "test_tokens": test.numel(), "perplexity": quality, **details}
    Path(args.quality).parent.mkdir(parents=True, exist_ok=True)
    Path(args.quality).write_text(json.dumps(report, indent=2))
    print("  perplexity  " + "  ".join(f"{k}={v:.3f}" for k, v in quality.items()) + f"  -> {args.quality}")
