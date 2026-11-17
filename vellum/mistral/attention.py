"""Attention math shared by the engine's rungs and AWQ calibration.

Tensors are (batch, positions, heads, head_dim) in and (batch, positions, heads * head_dim) out.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right


def allowed(i, j, window):
    """Query position i may attend key position j iff i - window < j <= i (the paper's SWA)."""
    return (j <= i) & (j > i - window)


def notebook_mask(n, window, like):
    """The notebook's prefill mask, built the notebook's way: 0 = attend, -inf = blocked."""
    t = torch.full((n, n), dtype=like.dtype, fill_value=1, device=like.device)
    return torch.log(torch.triu(torch.tril(t, diagonal=0), diagonal=-(window - 1)))


def notebook_attention(q, k, v, mask, scale):
    """The notebook's attention: matmul, additive mask, softmax, matmul."""
    b, n = q.shape[:2]
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    if mask is not None:
        scores += mask
    o = torch.matmul(F.softmax(scores, dim=-1).type_as(q), v)
    return o.transpose(1, 2).reshape(b, n, -1)


def sdpa_prefill(q, k, v, window):
    """Fused attention for a prompt that starts at position 0 (k, v already repeated per head)."""
    b, n = q.shape[:2]
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if n <= window:  # the window cannot bite yet: plain causal, which the flash kernel does natively
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        i = torch.arange(n, device=q.device)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed(i[:, None], i[None, :], window))
    return o.transpose(1, 2).reshape(b, n, -1)


def sdpa_chunk(q, k, v, start, lo, window):
    """Fused attention for a prefill chunk at positions start.. over cached keys lo..start + n."""
    b, n = q.shape[:2]
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    keys = k.shape[2]
    if lo == 0 and start + n <= window:  # no window yet: causal, aligned to the chunk's last query
        mask = causal_lower_right(n, keys)
    else:
        i = torch.arange(start, start + n, device=q.device)[:, None]
        j = torch.arange(lo, lo + keys, device=q.device)[None, :]
        mask = allowed(i, j, window)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return o.transpose(1, 2).reshape(b, n, -1)
