"""Fixed-batch-size benchmarks with no HTTP server in the loop (GPU box only).

    hf    Hugging Face Transformers `model.generate()` -- the "plain PyTorch" baseline.
          Static batching: the whole batch prefills together and decodes in lockstep.
    vllm  vLLM's offline `LLM.generate()` with the same batch of prompts, so the
          difference is purely the engine (PagedAttention, CUDA graphs, fused kernels).

Both backends are measured the same way, per batch size:
    prefill  = time for generate(max_new_tokens=1)            -> reported as TTFT
    total    = time for generate(max_new_tokens=output_len)   -> E2E
    TPOT     = (total - prefill) / (output_len - 1)
    tok/s    = batch_size * output_len / total

Prompts are random token IDs of exactly `input_len`, so there is no padding and
no tokenizer variance between backends.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _summary(label, model, bs, input_len, output_len, prefill_s, total_s, failed=False) -> dict:
    s = {
        "label": label, "model": model, "mode": "batch_size", "level": bs,
        "input_len": input_len, "output_len": output_len, "shared_prefix_len": 0,
        "num_requests": bs, "num_ok": 0 if failed else bs, "num_failed": bs if failed else 0,
    }
    if failed:
        s.update({k: None for k in ("duration_s", "request_throughput", "output_token_throughput",
                                    "total_token_throughput")})
        for name in ("ttft", "tpot", "itl", "e2e"):
            for p in ("mean", "p50", "p90", "p99"):
                s[f"{p}_{name}_ms"] = None
        return s
    tpot = (total_s - prefill_s) / max(1, output_len - 1)
    s.update({
        "duration_s": total_s,
        "request_throughput": bs / total_s,
        "output_token_throughput": bs * output_len / total_s,
        "total_token_throughput": bs * (input_len + output_len) / total_s,
    })
    # Every request in a static batch finishes together, so percentiles collapse to one value.
    for name, v in (("ttft", prefill_s), ("tpot", tpot), ("itl", tpot), ("e2e", total_s)):
        for p in ("mean", "p50", "p90", "p99"):
            s[f"{p}_{name}_ms"] = v * 1e3
    return s


def run_hf(args) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    model.eval()
    vocab = tok.vocab_size  # excludes added/special tokens
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def timed(ids, n_new):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                           min_new_tokens=n_new, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    out = []
    for bs in args.batch_sizes:
        ids = torch.randint(0, vocab, (bs, args.input_len), generator=gen).to("cuda")
        try:
            timed(ids, 4)  # warmup
            prefill = statistics.median(timed(ids, 1) for _ in range(args.repeats))
            total = statistics.median(timed(ids, args.output_len) for _ in range(args.repeats))
            s = _summary(args.label, args.model, bs, args.input_len, args.output_len, prefill, total)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            s = _summary(args.label, args.model, bs, args.input_len, args.output_len, 0, 0, failed=True)
            s["sample_errors"] = ["CUDA OOM"]
        out.append(s)
        _print(s)
        if s["num_failed"]:
            break
    return out


def run_vllm(args) -> list[dict]:
    import random

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="half", quantization=args.quantization,
              max_model_len=args.input_len + args.output_len + 16,
              max_num_seqs=max(args.batch_sizes), enable_prefix_caching=False,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(args.seed)

    def timed(prompts, n_new):
        sp = SamplingParams(max_tokens=n_new, min_tokens=n_new, ignore_eos=True, temperature=0.0)
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t0

    out = []
    for bs in args.batch_sizes:
        prompts = [{"prompt_token_ids": [rng.randrange(vocab) for _ in range(args.input_len)]}
                   for _ in range(bs)]
        timed(prompts, 4)  # warmup
        prefill = statistics.median(timed(prompts, 1) for _ in range(args.repeats))
        total = statistics.median(timed(prompts, args.output_len) for _ in range(args.repeats))
        s = _summary(args.label, args.model, bs, args.input_len, args.output_len, prefill, total)
        out.append(s)
        _print(s)
    return out


def _print(s: dict) -> None:
    if s["num_failed"]:
        print(f"  bs={s['level']:<4} FAILED ({s.get('sample_errors')})")
        return
    print(f"  bs={s['level']:<4} out_tok/s={s['output_token_throughput']:8.1f}  "
          f"prefill={s['p50_ttft_ms']:7.1f}ms  TPOT={s['p50_tpot_ms']:6.1f}ms")


def add_parser(sub) -> None:
    p = sub.add_parser("offline", help="fixed-batch benchmark: HF Transformers baseline or offline vLLM")
    p.add_argument("--backend", choices=["hf", "vllm"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--experiment", default="engine")
    p.add_argument("--quantization", default=None, help="vLLM only, e.g. awq_marlin; usually auto-detected")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")


def main(args: argparse.Namespace) -> None:
    print(f"[{args.label}] backend={args.backend} model={args.model}")
    levels = run_hf(args) if args.backend == "hf" else run_vllm(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"label": args.label, "experiment": args.experiment, "model": args.model,
            "backend": args.backend, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "server_args": f"offline {args.backend}" + (f" quantization={args.quantization}" if args.quantization else "")}
    path = out_dir / f"{args.label}.json"
    path.write_text(json.dumps({"meta": meta, "levels": levels}, indent=2))
    print(f"  -> {path}")
