"""Time every 4-bit matmul path on this GPU, per Mistral 7B layer shape and row count.

    python scripts/bench_int4.py --out results/NVIDIA-L4/int4_kernels.json

    bf16       cuBLAS on a full-precision copy of the weight
    tinygemm   PyTorch's fused int4 kernel (rung awq)
    dense      the weight rebuilt in one fused kernel, then the bf16 GEMM
    triton     the Triton kernels in vellum/mistral/w4a16.py, every config (rung triton)

Each kernel is checked against the dequantized weight first. Every path cycles through copies of
its weight that overflow L2, so each call streams it from memory as decode does; inputs of up to
64 rows are timed as CUDA-graph replays.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.nn import functional as F

from vellum.mistral import w4a16
from vellum.mistral.awq import (
    GROUP, ROWS, QuantLinear, candidates, dequantize, gpu_seconds, path_seconds, unpack_nibbles, weight_copies,
)

# (out, in) of Mistral 7B's linears, and how many of each one decode step runs per block
SHAPES = {(4096, 4096): 2, (1024, 4096): 2, (14336, 4096): 2, (4096, 14336): 1}  # wq wo, wk wv, w1 w3, w2
LAYERS = 32


def rel_err(got, want):
    return ((got.float() - want).norm() / want.norm()).item()


def bench_shape(out_f: int, in_f: int, rows, log) -> dict:
    torch.manual_seed(0)
    layer = QuantLinear.from_weight(torch.randn(out_f, in_f, device="cuda") * 0.02)
    reference = dequantize(unpack_nibbles(layer.nibbles), layer.sz)
    packed = torch.ops.aten._convert_weight_to_int4pack(layer.nibbles, 8)

    def tinygemm(x, packed=packed, sz=layer.sz):
        return torch.ops.aten._weight_int4pack_mm(x, packed, GROUP, sz)

    # every path cycles through copies of its weight that overflow L2, so each call streams it
    weights = weight_copies(layer.nibbles, layer.sz)
    fulls = weight_copies(reference.to(torch.bfloat16))
    packs = [(torch.ops.aten._convert_weight_to_int4pack(n, 8), s) for n, s in weights]

    configs = [(k, c) for k, cfgs in w4a16.CONFIGS.items() for c in cfgs if w4a16.valid(k, c, out_f, in_f)]
    broken = {}
    for kernel, cfg in configs:  # correctness first, at a row count that leaves partial tiles
        x = torch.randn(11 if kernel == "gemv" else 43, in_f, device="cuda", dtype=torch.bfloat16)
        try:
            err = rel_err(w4a16.matmul(kernel, cfg, x, layer.nibbles, layer.sz), x.float() @ reference.t())
        except Exception as e:  # noqa: BLE001 -- recorded, e.g. a tile too big for shared memory
            broken[w4a16.describe(kernel, cfg)] = f"{type(e).__name__}: {str(e)[:200]}"
            continue
        if not err < 1e-2:
            broken[w4a16.describe(kernel, cfg)] = f"relative error {err:.3g}"
    x = torch.randn(5, in_f, device="cuda", dtype=torch.bfloat16)
    tiny_err = rel_err(tinygemm(x), x.float() @ reference.t())

    result = {"rows": list(rows), "bf16": [], "tinygemm": [], "dense": [], "triton": [], "triton_config": [],
              "configs": {}, "broken": broken, "tinygemm_error": tiny_err,
              "copies": {"4-bit": len(weights), "bf16": len(fulls)}}
    for r in rows:
        x = torch.randn(r, in_f, device="cuda", dtype=torch.bfloat16)
        small = r <= 64

        def seconds(fn):
            return gpu_seconds(fn, graph=small, calls=20 if small else 3)

        result["bf16"].append(seconds([lambda w=w: F.linear(x, w) for (w,) in fulls]))
        result["tinygemm"].append(seconds([lambda p=p, s=s: tinygemm(x, p, s) for p, s in packs])
                                  if r <= 1024 else None)
        result["dense"].append(path_seconds(weights, "dense", {}, x))
        best = (float("inf"), None)
        for kernel, cfg in candidates(out_f, in_f, r, w4a16.CONFIGS):
            name = w4a16.describe(kernel, cfg)
            if name in broken:
                continue
            t = path_seconds(weights, kernel, cfg, x)
            result["configs"].setdefault(name, {})[r] = t
            best = min(best, (t, name))
        result["triton"].append(best[0])
        result["triton_config"].append(best[1])
        tiny = result["tinygemm"][-1]
        log(f"  {out_f}x{in_f} {r:5d} rows: bf16 {result['bf16'][-1] * 1e6:9.1f} us"
            f"  tinygemm {tiny * 1e6 if tiny else float('nan'):9.1f}  dense {result['dense'][-1] * 1e6:9.1f}"
            f"  triton {best[0] * 1e6:9.1f} ({best[1]})")
    return result


def decode_estimate(results: dict, rows: int) -> dict:
    """Seconds of linear-layer time per decode step at `rows` sequences, per path: every shape's
    time times how often a block runs it, times 32 blocks."""
    out = {}
    for path in ("bf16", "tinygemm", "dense", "triton"):
        total = 0.0
        for (o, i), count in SHAPES.items():
            res = results[f"{o}x{i}"]
            t = res[path][res["rows"].index(rows)]
            if t is None:
                total = None
                break
            total += count * t
        out[path] = None if total is None else total * LAYERS
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="write the timings here as JSON")
    parser.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    args = parser.parse_args(argv)
    if not w4a16.available():
        raise SystemExit("needs a CUDA GPU and Triton")
    import triton

    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, triton {triton.__version__}", flush=True)
    results = {f"{o}x{i}": bench_shape(o, i, args.rows, lambda s: print(s, flush=True)) for o, i in SHAPES}
    decode = {r: decode_estimate(results, r) for r in args.rows if r <= 64}
    for r, paths in decode.items():
        print(f"linears per decode step at {r:2d} rows: "
              + "  ".join(f"{p} {t * 1e3:6.2f} ms" for p, t in paths.items() if t is not None), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
            "python": platform.python_version(), "shapes": results, "decode_linears_seconds": decode,
        }, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
