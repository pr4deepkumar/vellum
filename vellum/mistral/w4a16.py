"""W4A16 Triton kernels: y = x @ W.T with W as the 4-bit codes awq.py writes, dequantized in
registers, so every byte read from memory is two weights.

Format: uint8 (out, in / 2) nibbles, even input column in the high nibble, and bf16
(in / 128, out, 2) [scale, offset] pairs: w = (q - 8) * scale + offset.

    gemv     CUDA cores, 1-8 input rows per program: decode, where the matmul is a weight stream
    gemm_t   tensor cores on y.T = W @ x.T, so the weight tile is the large operand and the few
             input rows, padded to 16, the small one: small batches
    gemm     tensor cores on y = x @ W.T: larger inputs

Each loop step stays inside one 128-wide quantization group (one scale and offset per output)
and multiplies the high nibbles by x's even columns and the low nibbles by its odd ones, so the
codes are never interleaved back into input order. awq.tune() picks kernel and tile per layer
shape and row count. Without Triton (CPU-only installs), `available()` is False.
"""

from __future__ import annotations

import itertools

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs: QuantLinear uses its dense path
    triton = tl = None

GROUP = 128  # weights per [scale, offset] pair; awq.GROUP
HALF = GROUP // 2  # bytes of codes per group

# The launch configs scripts/bench_int4.py searches. gemv's tile is BLOCK_M rows x BLOCK_N
# outputs x `groups` quantization groups per loop step; gemm's and gemm_t's are BLOCK_M rows x
# BLOCK_N outputs x BLOCK_KH bytes (2 * BLOCK_KH inputs). awq.tune() times only the TUNE_ lists.
GEMV_CONFIGS = tuple(
    dict(block_m=1, block_n=n, groups=g, num_warps=w, num_stages=1)
    for n, g, w in itertools.product((16, 32, 64, 128), (1, 2, 4), (2, 4, 8))
) + tuple(
    dict(block_m=m, block_n=n, groups=g, num_warps=w, num_stages=1)
    for m, n, g, w in itertools.product((2, 4, 8), (16, 32), (1, 2, 4), (4, 8))
)
GEMM_T_CONFIGS = tuple(
    dict(block_m=m, block_n=n, block_kh=k, num_warps=w, num_stages=s)
    for m, n, k, w, s in itertools.product((16, 32), (32, 64, 128), (32, 64), (4,), (3, 4))
)
GEMM_CONFIGS = tuple(
    dict(block_m=m, block_n=n, block_kh=k, num_warps=w, num_stages=s)
    for m, n, k, w, s in (
        (16, 32, 64, 4, 3), (16, 64, 64, 4, 3), (16, 128, 64, 4, 3), (16, 64, 32, 4, 4),
        (16, 128, 32, 4, 4), (16, 256, 32, 8, 3),
        (32, 64, 64, 4, 3), (32, 128, 64, 4, 3), (32, 128, 32, 4, 4),
        (64, 64, 64, 4, 3), (64, 128, 64, 4, 3), (64, 128, 32, 4, 4), (64, 256, 32, 8, 3),
        (128, 64, 64, 4, 3), (128, 128, 32, 8, 3), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3),
    )
)
CONFIGS = {"gemv": GEMV_CONFIGS, "gemm_t": GEMM_T_CONFIGS, "gemm": GEMM_CONFIGS}
# The configs that won somewhere on an L4 across Mistral 7B's four layer shapes, from 1 to 8,192
# rows (scripts/bench_int4.py); awq.tune() picks among these at load.
TUNE = {
    "gemv": tuple(dict(block_m=m, block_n=16, groups=g, num_warps=w, num_stages=1)
                  for m, g, w in ((1, 4, 8), (1, 1, 2), (1, 2, 4), (2, 1, 4), (2, 1, 8), (4, 1, 4))),
    "gemm_t": tuple(dict(block_m=16, block_n=n, block_kh=64, num_warps=4, num_stages=3) for n in (64, 128)),
    "gemm": tuple(dict(block_m=m, block_n=n, block_kh=k, num_warps=w, num_stages=s) for m, n, k, w, s in (
        (16, 32, 64, 4, 3), (16, 64, 64, 4, 3), (16, 128, 64, 4, 3), (32, 128, 64, 4, 3),
        (32, 128, 32, 4, 4), (64, 128, 64, 4, 3), (64, 128, 32, 4, 4), (64, 256, 32, 8, 3))),
}


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def valid(kernel: str, cfg: dict, out_features: int, in_features: int) -> bool:
    """Whether a config can run this shape: gemv steps whole groups, the gemms stay inside one."""
    if kernel == "gemv":
        return (in_features // GROUP) % cfg["groups"] == 0
    return HALF % cfg["block_kh"] == 0


def describe(kernel: str, cfg: dict) -> str:
    if kernel == "gemv":
        return f"gemv {cfg['block_m']}x{cfg['block_n']}x{cfg['groups']}g w{cfg['num_warps']}"
    return (f"{kernel} {cfg['block_m']}x{cfg['block_n']}x{2 * cfg['block_kh']} "
            f"w{cfg['num_warps']} s{cfg['num_stages']}")


if triton is not None:

    @triton.jit
    def _gemv_kernel(x_ptr, nib_ptr, sz_ptr, y_ptr, M, N, KH, stride_x, stride_y,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUPS: tl.constexpr, HALF: tl.constexpr):
        # row tiles vary fastest, so the programs running together share a weight tile in L2
        rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mmask = rm < M
        nmask = rn < N
        rj = tl.arange(0, HALF)
        if BLOCK_M == 1:
            acc = tl.zeros((BLOCK_N, HALF), dtype=tl.float32)
        else:  # each weight is loaded and dequantized once, then used by every row of the tile
            acc = tl.zeros((BLOCK_M, BLOCK_N, HALF), dtype=tl.float32)
        for start in range(0, KH, HALF * GROUPS):
            for i in tl.static_range(GROUPS):
                kh = start + i * HALF
                g = kh // HALF
                b = tl.load(nib_ptr + rn[:, None] * KH + (kh + rj)[None, :], mask=nmask[:, None], other=0)
                scale = tl.load(sz_ptr + (g * N + rn) * 2, mask=nmask, other=0.0).to(tl.float32)
                offset = tl.load(sz_ptr + (g * N + rn) * 2 + 1, mask=nmask, other=0.0).to(tl.float32)
                w_hi = ((b >> 4).to(tl.float32) - 8.0) * scale[:, None] + offset[:, None]
                w_lo = ((b & 15).to(tl.float32) - 8.0) * scale[:, None] + offset[:, None]
                x_at = x_ptr + rm[:, None] * stride_x + ((kh + rj) * 2)[None, :]
                x_even = tl.load(x_at, mask=mmask[:, None], other=0.0).to(tl.float32)
                x_odd = tl.load(x_at + 1, mask=mmask[:, None], other=0.0).to(tl.float32)
                if BLOCK_M == 1:
                    acc += w_hi * x_even + w_lo * x_odd
                else:
                    acc += w_hi[None, :, :] * x_even[:, None, :] + w_lo[None, :, :] * x_odd[:, None, :]
        if BLOCK_M == 1:
            y = tl.sum(acc, axis=1)[None, :]
        else:
            y = tl.sum(acc, axis=2)
        mask = mmask[:, None] & nmask[None, :]
        tl.store(y_ptr + rm[:, None] * stride_y + rn[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _gemm_t_kernel(x_ptr, nib_ptr, sz_ptr, y_ptr, M, N, KH, stride_x, stride_y,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KH: tl.constexpr, HALF: tl.constexpr):
        rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        rj = tl.arange(0, BLOCK_KH)
        mmask = rm < M
        nmask = rn < N
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)  # y.T
        for kh in range(0, KH, BLOCK_KH):
            g = kh // HALF  # BLOCK_KH divides HALF: the step sits inside one group
            b = tl.load(nib_ptr + rn[:, None] * KH + (kh + rj)[None, :], mask=nmask[:, None], other=0)
            scale = tl.load(sz_ptr + (g * N + rn) * 2, mask=nmask, other=0.0).to(tl.float32)
            offset = tl.load(sz_ptr + (g * N + rn) * 2 + 1, mask=nmask, other=0.0).to(tl.float32)
            w_hi = ((b >> 4).to(tl.float32) - 8.0) * scale[:, None] + offset[:, None]
            w_lo = ((b & 15).to(tl.float32) - 8.0) * scale[:, None] + offset[:, None]
            x_at = x_ptr + rm[None, :] * stride_x + ((kh + rj) * 2)[:, None]  # x.T: (BLOCK_KH, BLOCK_M)
            x_even = tl.load(x_at, mask=mmask[None, :], other=0.0)
            x_odd = tl.load(x_at + 1, mask=mmask[None, :], other=0.0)
            acc = tl.dot(w_hi.to(x_even.dtype), x_even, acc=acc)
            acc = tl.dot(w_lo.to(x_odd.dtype), x_odd, acc=acc)
        mask = nmask[:, None] & mmask[None, :]
        tl.store(y_ptr + rm[None, :] * stride_y + rn[:, None], acc.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _gemm_kernel(x_ptr, nib_ptr, sz_ptr, y_ptr, M, N, KH, stride_x, stride_y,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KH: tl.constexpr,
                     GROUP_M: tl.constexpr, HALF: tl.constexpr):
        # launch order: GROUP_M row tiles take turns on each weight tile while it is in L2
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BLOCK_M)
        num_n = tl.cdiv(N, BLOCK_N)
        per_band = GROUP_M * num_n
        first_m = (pid // per_band) * GROUP_M
        band_m = min(num_m - first_m, GROUP_M)
        pid_m = first_m + (pid % per_band) % band_m
        pid_n = (pid % per_band) // band_m
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rj = tl.arange(0, BLOCK_KH)
        mmask = rm < M
        nmask = rn < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kh in range(0, KH, BLOCK_KH):
            g = kh // HALF  # BLOCK_KH divides HALF: the step sits inside one group
            b = tl.load(nib_ptr + rn[None, :] * KH + (kh + rj)[:, None], mask=nmask[None, :], other=0)
            scale = tl.load(sz_ptr + (g * N + rn) * 2, mask=nmask, other=0.0).to(tl.float32)
            offset = tl.load(sz_ptr + (g * N + rn) * 2 + 1, mask=nmask, other=0.0).to(tl.float32)
            w_hi = ((b >> 4).to(tl.float32) - 8.0) * scale[None, :] + offset[None, :]
            w_lo = ((b & 15).to(tl.float32) - 8.0) * scale[None, :] + offset[None, :]
            x_at = x_ptr + rm[:, None] * stride_x + ((kh + rj) * 2)[None, :]
            x_even = tl.load(x_at, mask=mmask[:, None], other=0.0)
            x_odd = tl.load(x_at + 1, mask=mmask[:, None], other=0.0)
            acc = tl.dot(x_even, w_hi.to(x_even.dtype), acc=acc)
            acc = tl.dot(x_odd, w_lo.to(x_odd.dtype), acc=acc)
        mask = mmask[:, None] & nmask[None, :]
        tl.store(y_ptr + rm[:, None] * stride_y + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)


def gemv(x: torch.Tensor, nibbles: torch.Tensor, sz: torch.Tensor, block_m: int = 1, block_n: int = 16,
         groups: int = 4, num_warps: int = 8, num_stages: int = 1) -> torch.Tensor:
    m, k = x.shape
    n = nibbles.shape[0]
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    _gemv_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        x, nibbles, sz, y, m, n, k // 2, x.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, GROUPS=groups, HALF=HALF, num_warps=num_warps, num_stages=num_stages)
    return y


def gemm_t(x: torch.Tensor, nibbles: torch.Tensor, sz: torch.Tensor, block_m: int = 16, block_n: int = 64,
           block_kh: int = 64, num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    m, k = x.shape
    n = nibbles.shape[0]
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    _gemm_t_kernel[(triton.cdiv(n, block_n), triton.cdiv(m, block_m))](
        x, nibbles, sz, y, m, n, k // 2, x.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_KH=block_kh, HALF=HALF, num_warps=num_warps, num_stages=num_stages)
    return y


def gemm(x: torch.Tensor, nibbles: torch.Tensor, sz: torch.Tensor, block_m: int = 16, block_n: int = 64,
         block_kh: int = 64, num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    m, k = x.shape
    n = nibbles.shape[0]
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _gemm_kernel[grid](
        x, nibbles, sz, y, m, n, k // 2, x.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_KH=block_kh, GROUP_M=8, HALF=HALF,
        num_warps=num_warps, num_stages=num_stages)
    return y


KERNELS = {"gemv": gemv, "gemm_t": gemm_t, "gemm": gemm}


def matmul(kernel: str, cfg: dict, x: torch.Tensor, nibbles: torch.Tensor, sz: torch.Tensor) -> torch.Tensor:
    """x (rows, in) with unit stride along `in`, by the 4-bit weight; (rows, out) in x's dtype."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    return KERNELS[kernel](x, nibbles, sz, **cfg)
