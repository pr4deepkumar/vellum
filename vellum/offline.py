"""Fixed-batch-size benchmarks with no HTTP server in the loop (GPU box only).

    hf    Hugging Face Transformers `model.generate()` -- the "plain PyTorch" baseline.
          Static batching: the whole batch prefills together and decodes in lockstep.
    vllm  vLLM's offline `LLM.generate()` with the same batch of prompts, so the
          difference is purely the engine (PagedAttention, CUDA graphs, fused kernels).
    scratch
          The from-scratch Mistral (vellum.mistral), at one rung of its optimization
          ladder (--rung), on the same random prompts as hf.

All backends are measured the same way, per batch size:
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

# vellum.mistral.LADDER, repeated here so the CLI parses without torch installed.
SCRATCH_RUNGS = ("naive", "lastlogit", "nosync", "gqa", "sdpa", "chunked", "graph", "awq", "triton")


def _summary(label, model, bs, input_len, output_len, prefill_s, total_s, failed=False,
             peak_mem_gib=None) -> dict:
    s = {
        "label": label, "model": model, "mode": "batch_size", "level": bs,
        "input_len": input_len, "output_len": output_len, "shared_prefix_len": 0,
        "num_requests": bs, "num_ok": 0 if failed else bs, "num_failed": bs if failed else 0,
        "peak_mem_gib": None if failed else peak_mem_gib,  # weights + cache + activations at peak
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
        torch.cuda.reset_peak_memory_stats()
        try:
            timed(ids, 4)  # warmup
            prefill = statistics.median(timed(ids, 1) for _ in range(args.repeats))
            total = statistics.median(timed(ids, args.output_len) for _ in range(args.repeats))
            s = _summary(args.label, args.model, bs, args.input_len, args.output_len, prefill, total,
                         peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30)
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


def run_scratch(args) -> list[dict]:
    import torch

    from .mistral import Engine, load_model

    device = torch.device(args.device)
    on_gpu = device.type == "cuda"
    # fp16 like the other backends (an AWQ checkpoint loads in bf16, which its kernels need); the
    # CPU path exists for tests, where fp16 matmuls crawl
    single_copy = getattr(args, "awq_single_copy", False)
    if single_copy and args.rung != "triton":
        raise SystemExit("--awq-single-copy applies to rung triton: rung awq has only PyTorch's int4 kernel")
    model = load_model(args.model, device=device, dtype=torch.float16 if on_gpu else torch.float32,
                       triton=args.rung == "triton", int4pack=not single_copy)
    engine = Engine(model, rung=args.rung)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)  # run_hf's prompts, when vocab sizes match

    def timed(ids, n_new):
        if on_gpu:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        engine.generate(ids, n_new)
        if on_gpu:
            torch.cuda.synchronize(device)
        return time.perf_counter() - t0

    out = []
    for bs in args.batch_sizes:
        ids = torch.randint(0, model.args.vocab_size, (bs, args.input_len), generator=gen).to(device)
        if on_gpu:
            torch.cuda.reset_peak_memory_stats(device)
        try:
            engine.setup(bs, args.input_len + args.output_len)  # KV cache sized to this level
            timed(ids, min(4, args.output_len))  # warmup
            prefill = statistics.median(timed(ids, 1) for _ in range(args.repeats))
            total = statistics.median(timed(ids, args.output_len) for _ in range(args.repeats))
            peak = torch.cuda.max_memory_allocated(device) / 2**30 if on_gpu else None
            s = _summary(args.label, args.model, bs, args.input_len, args.output_len, prefill, total,
                         peak_mem_gib=peak)
        except torch.cuda.OutOfMemoryError:
            s = _summary(args.label, args.model, bs, args.input_len, args.output_len, 0, 0, failed=True)
            s["sample_errors"] = ["CUDA OOM"]
        finally:
            engine.teardown()
            if on_gpu:
                torch.cuda.empty_cache()
        out.append(s)
        _print(s)
        if s["num_failed"]:
            break
    return out


def _print(s: dict) -> None:
    if s["num_failed"]:
        print(f"  bs={s['level']:<4} FAILED ({s.get('sample_errors')})")
        return
    print(f"  bs={s['level']:<4} out_tok/s={s['output_token_throughput']:8.1f}  "
          f"prefill={s['p50_ttft_ms']:7.1f}ms  TPOT={s['p50_tpot_ms']:6.1f}ms")


def add_parser(sub) -> None:
    p = sub.add_parser("offline", help="fixed-batch benchmark: HF Transformers, offline vLLM, "
                                          "or the from-scratch Mistral engine")
    p.add_argument("--backend", choices=["hf", "vllm", "scratch"], required=True)
    p.add_argument("--model", required=True,
                   help="Hub repo id; scratch also takes a local folder holding consolidated.safetensors "
                        "+ params.json (e.g. the notebook's WEIGHTS_DIR)")
    p.add_argument("--label", required=True)
    p.add_argument("--experiment", default="engine")
    p.add_argument("--quantization", default=None, help="vLLM only, e.g. awq_marlin; usually auto-detected")
    p.add_argument("--rung", choices=SCRATCH_RUNGS, default="graph",
                   help="scratch only: rung of the optimization ladder; each keeps every rung before it")
    p.add_argument("--device", default="cuda", help="scratch only; cpu is the test path")
    p.add_argument("--awq-single-copy", action="store_true",
                   help="rung triton only: never build the second copy of the 4-bit weights that "
                        "PyTorch's int4 kernel reads. On Mistral 7B on an L4: about 3.1 GiB less GPU "
                        "memory from load on, decode at batch 1-8 about 15-25%% slower, larger batches "
                        "unaffected")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")


def main(args: argparse.Namespace) -> None:
    server_args = f"offline {args.backend}"
    if args.quantization:
        server_args += f" quantization={args.quantization}"
    if args.backend == "scratch":
        server_args += f" rung={args.rung}" + (" single-copy" if args.awq_single_copy else "")
    print(f"[{args.label}] {server_args} model={args.model}")
    levels = {"hf": run_hf, "vllm": run_vllm, "scratch": run_scratch}[args.backend](args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"label": args.label, "experiment": args.experiment, "model": args.model,
            "backend": args.backend, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "server_args": server_args}
    path = out_dir / f"{args.label}.json"
    path.write_text(json.dumps({"meta": meta, "levels": levels}, indent=2))
    print(f"  -> {path}")
