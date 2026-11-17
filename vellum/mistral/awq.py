"""AWQ: 4-bit weights, scaled by how large each input channel's activations are.

Lin et al., 2023, "AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration" (arXiv:2306.00978).

A decode step mostly reads weights, so holding them in 4 bits instead of 16 cuts the bytes per
token about 3.5x. Rounding to 16 levels costs the most where activations are large, so before
quantizing, AWQ multiplies each linear's input channels by s = mean|x|^alpha and divides the op
that feeds them by s. That is exact in full precision, and it enlarges the weights that matter
most, so rounding hurts them less. alpha is searched per group of linears on calibration text.

    quantize       asymmetric 4 bits, one scale and offset per 128 inputs:
                   w = (q - 8) * scale + offset, held as nibble-packed codes plus bf16 pairs
    QuantLinear    a 4-bit nn.Linear over those codes. Its matmul paths: PyTorch's int4 kernel
                   (on its own copy of the codes), a dense GEMM on the weight rebuilt in one
                   fused kernel, and the Triton W4A16 kernels in w4a16.py. tune() times the
                   allowed paths per layer shape and row count; forward() follows the fastest.
    awq_quantize   calibrate, search and fold the scales block by block, then quantize. With
                   search=False it is plain round-to-nearest (RTN), the baseline AWQ must beat.

Simplifications next to the reference implementation: scales are searched against each linear's
own output error rather than the whole block's, there is no weight-clipping search, and wo keeps
unscaled inputs, because grouped-query attention leaves no exact place to fold its scales
(AutoAWQ skips it for the same reason).
"""

from __future__ import annotations

import math
import statistics

import torch
from torch import nn
from torch.nn import functional as F

from . import w4a16
from .attention import sdpa_prefill
from .model import RMSNorm, Transformer, apply_rotary_emb, repeat_kv

GROUP = w4a16.GROUP  # weights per (scale, offset) pair, along the input dimension
ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)  # row counts tune() times
GEMV_ROWS = 8  # tune() tries the gemv kernel up to this many rows
GEMM_T_ROWS = 64  # and gemm_t up to this many
INT4PACK_ROWS = 64  # and PyTorch's int4 kernel, which slows linearly with rows, up to this many
# (out, in) -> ((up to this many rows, kernel, launch config), ...), ascending; see tune()
_PLANS: dict[tuple[int, int], tuple[tuple[int, str, dict], ...]] = {}
LINEARS = (("attention", "wq"), ("attention", "wk"), ("attention", "wv"), ("attention", "wo"),
           ("feed_forward", "w1"), ("feed_forward", "w2"), ("feed_forward", "w3"))


# -- 4-bit format ----------------------------------------------------------------------


def quantize(w: torch.Tensor, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """An (out, in) weight as int32 codes in [0, 15] plus bf16 (in / group, out, 2) [scale, offset]."""
    out_f, in_f = w.shape
    wg = w.float().reshape(out_f, in_f // group, group)
    lo, hi = wg.amin(-1), wg.amax(-1)
    scale = ((hi - lo) / 15).clamp(min=1e-6).to(torch.bfloat16)
    offset = (lo + 8 * scale.float()).to(torch.bfloat16)
    # codes from the rounded scale and offset, so they agree exactly with every reader
    base = offset.float() - 8 * scale.float()
    q = ((wg - base[..., None]) / scale.float()[..., None]).round().clamp(0, 15).to(torch.int32)
    return q.reshape(out_f, in_f), torch.stack([scale, offset], -1).transpose(0, 1).contiguous()


def dequantize(q: torch.Tensor, sz: torch.Tensor, dtype=torch.float32, group: int = GROUP) -> torch.Tensor:
    """Codes and [scale, offset] pairs back to an (out, in) weight in `dtype`."""
    out_f, in_f = q.shape
    scale = sz[..., 0].t().to(dtype)[..., None]  # (out, in / group, 1)
    offset = sz[..., 1].t().to(dtype)[..., None]
    return ((q.view(out_f, in_f // group, group).to(dtype) - 8) * scale + offset).view(out_f, in_f)


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """int32 codes (out, in) as uint8 (out, in / 2), each even column in the high nibble."""
    return ((q[:, 0::2] << 4) | q[:, 1::2]).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.int32)
    return torch.stack([p >> 4, p & 15], dim=-1).view(p.shape[0], -1)


def _dense(nibbles: torch.Tensor, sz: torch.Tensor, dtype) -> torch.Tensor:
    """The weight rebuilt from nibble-packed codes: computed in fp32, rounded to `dtype` once."""
    out_f = nibbles.shape[0]
    q = torch.stack([nibbles >> 4, nibbles & 15], dim=-1).view(out_f, -1, GROUP).float()
    w = (q - 8) * sz[..., 0].t().float()[..., None] + sz[..., 1].t().float()[..., None]
    return w.view(out_f, -1).to(dtype)


_dense_fused = None  # torch.compile(_dense) on CUDA: one kernel instead of a full pass per op


def dense_weight(nibbles: torch.Tensor, sz: torch.Tensor, dtype) -> torch.Tensor:
    global _dense_fused
    if not nibbles.is_cuda:
        return _dense(nibbles, sz, dtype)
    if _dense_fused is None:
        _dense_fused = torch.compile(_dense, dynamic=False)
    return _dense_fused(nibbles, sz, dtype)


def route(plan, rows: int) -> tuple[str, dict]:
    """The (kernel, config) a tuned plan picks for `rows` rows: the entry for the smallest timed
    row count at or above it, or past the largest, the one that won there."""
    for upto, kernel, cfg in plan:
        if rows <= upto:
            return kernel, cfg
    return plan[-1][1], plan[-1][2]


def int4pack_layout(nibbles: torch.Tensor) -> torch.Tensor:
    """The codes in the tiled layout PyTorch's CUDA int4 kernel reads (since 2.5 it takes this
    nibble format as input)."""
    return torch.ops.aten._convert_weight_to_int4pack(nibbles, 8)


def int4pack_mm(x: torch.Tensor, packed: torch.Tensor, sz: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten._weight_int4pack_mm(x.contiguous(), packed, GROUP, sz)


class QuantLinear(nn.Module):
    """A bias-free linear layer over 4-bit codes (see quantize). `_int4pack`, a second copy of the
    codes in PyTorch's int4 layout, exists only where tune() found that kernel fastest."""

    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        if in_features % GROUP:
            raise ValueError(f"in_features must be a multiple of {GROUP}, got {in_features}")
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("nibbles", torch.zeros(out_features, in_features // 2, dtype=torch.uint8,
                                                    device=device))
        self.register_buffer("sz", torch.zeros(in_features // GROUP, out_features, 2, dtype=torch.bfloat16,
                                               device=device))
        self._int4pack = None  # set by tune() only if PyTorch's int4 kernel wins somewhere

    @classmethod
    def from_weight(cls, w: torch.Tensor) -> QuantLinear:
        q, sz = quantize(w)
        layer = cls(w.shape[1], w.shape[0], device=w.device)
        layer.nibbles.copy_(pack_nibbles(q))
        layer.sz.copy_(sz)
        return layer

    def dense_weight(self, dtype) -> torch.Tensor:
        return dense_weight(self.nibbles, self.sz, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Untuned shapes, and anything off CUDA, take the dense path."""
        plan = _PLANS.get((self.out_features, self.in_features)) if x.is_cuda else None
        if plan is not None:
            rows = x.numel() // self.in_features
            kernel, cfg = route(plan, rows)
            if kernel == "int4pack" and self._int4pack is not None:
                out = int4pack_mm(x.reshape(rows, self.in_features), self._int4pack, self.sz)
                return out.view(*x.shape[:-1], self.out_features)
            if kernel in w4a16.KERNELS:
                out = w4a16.matmul(kernel, cfg, x.reshape(rows, self.in_features), self.nibbles, self.sz)
                return out.view(*x.shape[:-1], self.out_features)
        return F.linear(x, self.dense_weight(x.dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits=4, group={GROUP}"


def make_quantized(model: Transformer, device=None) -> None:
    """Swap every block's linears for empty QuantLinears, to load a quantized checkpoint into."""
    for layer in model.layers:
        for parent_name, name in LINEARS:
            parent = getattr(layer, parent_name)
            old = getattr(parent, name)
            setattr(parent, name, QuantLinear(old.in_features, old.out_features, device=device))


def gpu_seconds(fns, graph: bool = True, calls: int = 20, repeats: int = 5) -> float:
    """Median GPU time of one call, cycling through `fns`. graph=True replays the calls from one
    CUDA graph, as the graph rung runs decode, so launch overhead does not count; graph=False
    runs them eagerly, which suits prefill-sized calls."""
    fns = [fns] if callable(fns) else list(fns)
    calls = max(calls, len(fns))

    def burst():
        for i in range(calls):
            fns[i % len(fns)]()

    for fn in fns:  # compile and allocate off-capture
        fn()
    torch.cuda.synchronize()
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            burst()
        torch.cuda.current_stream().wait_stream(side)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            burst()
        run = captured.replay
    else:
        run = burst
    run()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3 / calls)
    return statistics.median(times)


def weight_copies(*tensors: torch.Tensor, cache_multiple: int = 3) -> list[tuple[torch.Tensor, ...]]:
    """Clones of a weight's tensors totalling `cache_multiple` times this GPU's L2 cache.

    Timing one weight over and over lets it sit in L2 (a 4-bit Mistral linear is 2-29 MB, an
    L4's L2 48 MB), which decode never sees: it reads each layer from memory once per step.
    Cycling through the copies makes every timed call stream its weight too."""
    cache = torch.cuda.get_device_properties(tensors[0].device).L2_cache_size
    size = sum(t.numel() * t.element_size() for t in tensors)
    return [tuple(t.clone() for t in tensors) for _ in range(max(1, math.ceil(cache_multiple * cache / size)))]


def path_seconds(weights, kernel: str, cfg: dict, x: torch.Tensor) -> float:
    """GPU time of x (rows, in) times a 4-bit weight along one path. `weights` holds (nibbles, sz)
    copies from weight_copies, or (layout, sz) pairs for "int4pack"."""
    if kernel == "dense":
        fns = [lambda n=n, s=s: F.linear(x, dense_weight(n, s, x.dtype)) for n, s in weights]
    elif kernel == "int4pack":
        fns = [lambda p=p, s=s: int4pack_mm(x, p, s) for p, s in weights]
    else:
        fns = [lambda n=n, s=s: w4a16.matmul(kernel, cfg, x, n, s) for n, s in weights]
    small = x.shape[0] <= 64
    return gpu_seconds(fns, graph=small, calls=20 if small else 3)


def candidates(out_features: int, in_features: int, rows: int, configs=None):
    """The (kernel, config) pairs worth timing at `rows` rows, from w4a16.TUNE by default."""
    configs = w4a16.TUNE if configs is None else configs
    for kernel, cfgs in configs.items():
        for cfg in cfgs:
            if not w4a16.valid(kernel, cfg, out_features, in_features):
                continue
            m = cfg["block_m"]
            if kernel == "gemv" and not (rows <= GEMV_ROWS and m <= rows):
                continue
            if kernel == "gemm_t" and not (rows <= GEMM_T_ROWS and m <= max(16, rows)):
                continue
            if kernel == "gemm" and not min(64, rows // 16) <= m <= max(16, 2 * rows):
                continue  # row tiles neither far wider than the input nor tiny for it
            yield kernel, cfg


def _name(kernel: str, cfg: dict) -> str:
    return {"dense": "dense", "int4pack": "pytorch int4"}.get(kernel) or w4a16.describe(kernel, cfg)


def _check(layer: QuantLinear, kernel: str, cfg: dict, rows: int, reference: torch.Tensor) -> None:
    """The kernel against x @ the dequantized weight, at `rows` and at a partial last tile."""
    packed = int4pack_layout(layer.nibbles) if kernel == "int4pack" else None
    for r in sorted({rows, rows - rows // 3}):
        x = torch.randn(r, layer.in_features, device=reference.device, dtype=torch.bfloat16)
        if packed is not None:
            got = int4pack_mm(x, packed, layer.sz).float()
        else:
            got = w4a16.matmul(kernel, cfg, x, layer.nibbles, layer.sz).float()
        want = x.float() @ reference.t()
        err = ((got - want).norm() / want.norm()).item()
        if not err < 1e-2:
            raise RuntimeError(f"{_name(kernel, cfg)} disagrees with the dequantized "
                               f"{layer.out_features}x{layer.in_features} weight at {r} rows "
                               f"(relative error {err:.3g})")


def _tune_shape(layer: QuantLinear, rows, triton: bool, int4pack: bool) -> tuple[tuple[int, str, dict], ...]:
    reference = dequantize(unpack_nibbles(layer.nibbles), layer.sz)  # fp32, apart from every path
    weights = weight_copies(layer.nibbles, layer.sz)
    packs = [(int4pack_layout(n), s) for n, s in weights] if int4pack else None
    plan, dense_streak, tried, failed = [], 0, 0, []
    for r in rows:
        x = torch.randn(r, layer.in_features, device=layer.nibbles.device, dtype=torch.bfloat16)
        timed = [(path_seconds(weights, "dense", {}, x), "dense", {})]
        if dense_streak < 2:  # dense only gains with rows: once it has won twice, it keeps winning
            options = list(candidates(layer.out_features, layer.in_features, r)) if triton else []
            if packs and r <= INT4PACK_ROWS:
                options.append(("int4pack", {}))
            for kernel, cfg in options:
                tried += 1
                try:
                    timed.append((path_seconds(packs if kernel == "int4pack" else weights, kernel, cfg, x),
                                  kernel, cfg))
                except Exception as e:  # e.g. a tile that needs more shared memory than this GPU has
                    failed.append(e)
        # The GPU clocks down under the sweep's sustained load, so late candidates look slower.
        # Re-time the best of each kind (dense, PyTorch's int4, Triton) back to back in
        # A B C C B A order, which cancels a steady drift, and keep the lowest total.
        best: dict[str, tuple] = {}
        for seconds, kernel, cfg in timed:
            kind = kernel if kernel in ("dense", "int4pack") else "triton"
            if kind not in best or seconds < best[kind][0]:
                best[kind] = (seconds, kernel, cfg)
        finalists = [(kernel, cfg) for _, kernel, cfg in best.values()]
        totals = [0.0] * len(finalists)
        for i in [*range(len(finalists)), *reversed(range(len(finalists)))]:
            kernel, cfg = finalists[i]
            totals[i] += path_seconds(packs if kernel == "int4pack" else weights, kernel, cfg, x)
        kernel, cfg = finalists[min(range(len(finalists)), key=totals.__getitem__)]
        if kernel != "dense":
            _check(layer, kernel, cfg, r, reference)
        dense_streak = dense_streak + 1 if kernel == "dense" else 0
        if plan and plan[-1][1:] == (kernel, cfg):
            plan[-1] = (r, kernel, cfg)  # the same winner as the previous row count: widen it
        else:
            plan.append((r, kernel, cfg))
    if tried and len(failed) == tried:
        raise RuntimeError(f"no kernel ran on {layer.out_features}x{layer.in_features}") from failed[0]
    return tuple(plan)


_TUNED_WITH_TRITON = False  # whether the last tune() could pick the Triton kernels; see Engine


@torch.no_grad()
def tune(model: Transformer, rows=ROWS, triton: bool = True, int4pack: bool = True) -> dict[tuple[int, int], tuple]:
    """Per linear shape and row count, time the allowed paths, check the winner against the
    dequantized weight, and record it for forward(). Returns {} off CUDA.

    The dense rebuild is always allowed. triton=True adds the Triton kernels (rung `triton`).
    int4pack=True adds PyTorch's int4 kernel, whose layout is a second copy of the codes, kept
    only for shapes whose plan uses it. int4pack=False never builds that copy: on Mistral 7B on
    an L4, about 3.1 GiB less memory from load on, batch 1-8 decode 15-25% slower, larger
    batches unaffected."""
    global _TUNED_WITH_TRITON
    layers = [m for m in model.modules() if isinstance(m, QuantLinear) and m.nibbles.is_cuda]
    if not layers:
        return {}
    if not (triton or int4pack):
        raise ValueError("nothing to tune: allow the Triton kernels, PyTorch's int4 kernel, or both")
    if triton and not w4a16.available():
        raise RuntimeError("the Triton kernels need Triton, which ships with PyTorch's Linux CUDA wheels")
    shapes = {}
    for m in layers:
        shapes.setdefault((m.out_features, m.in_features), m)
    for shape, layer in shapes.items():
        _PLANS[shape] = _tune_shape(layer, rows, triton, int4pack)
    for m in layers:  # the second copy only where it pays
        uses = any(kernel == "int4pack" for _, kernel, _ in _PLANS[(m.out_features, m.in_features)])
        m._int4pack = int4pack_layout(m.nibbles) if uses else None
    _TUNED_WITH_TRITON = triton
    return {shape: _PLANS[shape] for shape in shapes}


def tuned_with_triton() -> bool:
    return _TUNED_WITH_TRITON


def describe_plan(plan) -> str:
    """e.g. 'pytorch int4 to 8 rows, gemm 16x64x128 w4 s3 to 64 rows, dense from 65 rows'"""
    parts, first = [], 1
    for i, (upto, kernel, cfg) in enumerate(plan):
        name = _name(kernel, cfg)
        parts.append(f"{name} from {first} rows" if i == len(plan) - 1 else f"{name} to {upto} rows")
        first = upto + 1
    return ", ".join(parts)


# -- AWQ -------------------------------------------------------------------------------


def _output_error(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, chunk: int = 4096) -> float:
    """Mean squared output error over the rows of x of quantizing w with its inputs scaled by s."""
    wf = w.float()
    delta = (dequantize(*quantize(wf * s)) / s - wf).to(x.dtype)
    total = 0.0
    for i in range(0, x.shape[0], chunk):
        total += (x[i : i + chunk] @ delta.t()).float().pow(2).sum().item()
    return total / (x.shape[0] * w.shape[0])


def search_scale(x: torch.Tensor, weights, n_grid: int = 20) -> torch.Tensor:
    """AWQ's input-channel scales for weights that all read x (rows, in): s = mean|x|^alpha with
    alpha on a grid over [0, 1), keeping the lowest total output error. alpha = 0 is plain RTN."""
    x_mean = x.abs().float().mean(0)
    best_err, best = math.inf, None
    for i in range(n_grid):
        s = x_mean.pow(i / n_grid).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        err = sum(_output_error(x, w, s) for w in weights)
        if err < best_err:
            best_err, best = err, s
    return best


def _fold(s: torch.Tensor, linears, prev) -> None:
    """Scale the linears' input columns by s and divide the op feeding them by s (exact)."""
    for lin in linears:
        lin.weight.mul_(s.to(lin.weight.dtype))
    if isinstance(prev, RMSNorm):
        prev.weight.div_(s.to(prev.weight.dtype))
    else:  # a linear whose output rows feed these inputs
        prev.weight.div_(s.to(prev.weight.dtype)[:, None])


def _attention(attn, x, cs):
    """A block's attention over whole calibration sequences, without touching the KV cache."""
    b, n, _ = x.shape
    hd = attn.args.head_dim
    q = apply_rotary_emb(attn.wq(x).view(b, n, attn.n_heads, hd), cs)
    k = apply_rotary_emb(attn.wk(x).view(b, n, attn.n_kv_heads, hd), cs)
    v = attn.wv(x).view(b, n, attn.n_kv_heads, hd)
    k, v = repeat_kv(k, v, attn.repeats)
    return attn.wo(sdpa_prefill(q, k, v, attn.args.sliding_window))


@torch.no_grad()
def awq_quantize(model: Transformer, calibration: torch.Tensor, search: bool = True, n_grid: int = 20,
                 log=None) -> None:
    """Quantize every block's linears in place. calibration: (samples, length) token ids."""
    h = model.tok_embeddings(calibration)
    cs = model.cosin_m_thetas[: calibration.shape[1]]
    for i, layer in enumerate(model.layers):
        a, f = layer.attention, layer.feed_forward
        if search:
            x = layer.attention_norm(h).flatten(0, 1)
            _fold(search_scale(x, [a.wq.weight, a.wk.weight, a.wv.weight], n_grid), [a.wq, a.wk, a.wv],
                  layer.attention_norm)
        h = h + _attention(a, layer.attention_norm(h), cs)
        if search:
            x = layer.ffn_norm(h).flatten(0, 1)
            _fold(search_scale(x, [f.w1.weight, f.w3.weight], n_grid), [f.w1, f.w3], layer.ffn_norm)
            x = layer.ffn_norm(h).flatten(0, 1)
            m = F.silu(f.w1(x)) * f.w3(x)
            _fold(search_scale(m, [f.w2.weight], n_grid), [f.w2], f.w3)
        h = h + f(layer.ffn_norm(h))  # the next block calibrates on this block's full-precision output
        for parent_name, name in LINEARS:
            parent = getattr(layer, parent_name)
            setattr(parent, name, QuantLinear.from_weight(getattr(parent, name).weight))
        if log:
            log(f"block {i + 1}/{len(model.layers)}")


# -- data and quality ------------------------------------------------------------------


def wikitext(split: str) -> str:
    """WikiText-2 (raw) as one string, joined the way perplexity evaluations usually join it."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                           repo_type="dataset")
    return "\n\n".join(pq.read_table(path).column("text").to_pylist())


def calibration_batch(ids: torch.Tensor, samples: int, length: int, seed: int = 0) -> torch.Tensor:
    """`samples` random windows of `length` tokens from a 1-D token stream."""
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - length, (samples,), generator=g).tolist()
    return torch.stack([ids[s : s + length] for s in starts])


@torch.no_grad()
def perplexity(model: Transformer, ids: torch.Tensor, window: int = 2048) -> float:
    """exp(mean next-token loss) over a 1-D token stream, in non-overlapping windows."""
    device = model.output.weight.device
    model.allocate_cache(1, window)
    total, count = 0.0, 0
    for s in range(0, ids.numel() - 1, window):
        chunk = ids[s : s + window + 1].to(device)
        logits = model(chunk[None, :-1], torch.arange(chunk.numel() - 1, device=device))
        total += F.cross_entropy(logits[0].float(), chunk[1:], reduction="sum").item()
        count += chunk.numel() - 1
    model.release_cache()
    return math.exp(total / count)
