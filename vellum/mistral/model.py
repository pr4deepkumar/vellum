"""Mistral 7B inference from scratch in plain PyTorch: the model notebooks/01_build_mistral.ipynb builds.

RoPE, grouped-query attention, a preallocated KV cache, RMSNorm and SwiGLU, computed exactly as
the notebook computes them (tests/test_notebook_model.py holds the two to identical outputs).
This is the `naive` rung of the engine's optimization ladder; engine.py makes it fast without
changing it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import nn

if TYPE_CHECKING:
    from mistral_common.tokens.tokenizers.base import Tokenizer

ROPE_POSITIONS = 128_000  # positions the RoPE table covers


@dataclass
class TransformerArgs:
    dim: int = 4096
    n_layers: int = 32
    head_dim: int = 128
    hidden_dim: int = 14336
    n_heads: int = 32
    n_kv_heads: int = 8
    norm_eps: float = 1e-05
    vocab_size: int = 32768
    rope_N: float = 1000000.0  # RoPE base; params.json calls it rope_theta
    max_seq_len: int = 16384
    # W from the paper: token i attends to j where i - W < j <= i. v0.3's params.json sets no
    # window (v0.3 was trained without one), so this is the paper's 4096; it only matters past it.
    sliding_window: int = 4096
    max_batch_size: int = 1

    @classmethod
    def from_params(cls, params: dict) -> TransformerArgs:
        """From Mistral's params.json: rope_theta becomes rope_N, keys this model lacks are ignored."""
        params = dict(params)
        if "rope_theta" in params:
            params["rope_N"] = params.pop("rope_theta")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in known})


def precompute_thetas(dim, end, rope_N):
    thetas = 1.0 / (rope_N ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    m_thetas = torch.outer(torch.arange(0, end), thetas)
    complexthetas = torch.polar(torch.ones_like(m_thetas), m_thetas)
    return torch.stack([complexthetas.real, complexthetas.imag], dim=-1)


def apply_rotary_emb(x, cosin_m_thetas):
    # x: (bs, seqlen, n_heads, head_dim), cosin_m_thetas: (seqlen, head_dim/2, 2)
    xreshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    cosin_m_thetas = cosin_m_thetas.view(1, xreshaped.size(1), 1, xreshaped.size(3), 2)
    x_rot = torch.stack(
        [
            xreshaped[..., 0] * cosin_m_thetas[..., 0] - xreshaped[..., 1] * cosin_m_thetas[..., 1],
            xreshaped[..., 1] * cosin_m_thetas[..., 0] + xreshaped[..., 0] * cosin_m_thetas[..., 1],
        ],
        -1,
    )
    return x_rot.flatten(3).type_as(x)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


class Attention(nn.Module):
    def __init__(self, args: TransformerArgs, device="cpu"):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        # plain tensors, not buffers: .to() won't move them, so they are placed up front
        shape = (args.max_batch_size, args.max_seq_len, self.n_kv_heads, args.head_dim)
        self.cache_k = torch.empty(shape, dtype=torch.float16, device=device)
        self.cache_v = torch.empty(shape, dtype=torch.float16, device=device)

    def forward(self, x, cosin_m_thetas, positions, mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)

        xq = apply_rotary_emb(xq, cosin_m_thetas)
        xk = apply_rotary_emb(xk, cosin_m_thetas)

        scatter_pos = positions[None, :, None, None].repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
        self.cache_k = self.cache_k.type(xk.dtype)
        self.cache_v = self.cache_v.type(xv.dtype)
        self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk)
        self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv)

        if positions.shape[0] > 1:
            # prefill
            key, value = repeat_kv(xk, xv, self.repeats)
        else:
            # decode - read everything so far from the cache
            cur_pos = positions[-1].item() + 1
            key, value = repeat_kv(self.cache_k[:bsz, :cur_pos], self.cache_v[:bsz, :cur_pos], self.repeats)

        query, key, value = xq.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        if mask is not None:
            scores += mask[None, None, ...]
        scores = nn.functional.softmax(scores, dim=-1).type_as(query)
        output = torch.matmul(scores, value)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        # SwiGLU
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: TransformerArgs, device="cpu"):
        super().__init__()
        self.attention = Attention(args, device=device)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, cosin_m_thetas, positions, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), cosin_m_thetas, positions, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    def __init__(self, args: TransformerArgs, device="cpu"):
        super().__init__()
        self.args = args
        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args, device=device) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.cosin_m_thetas = precompute_thetas(args.head_dim, ROPE_POSITIONS, args.rope_N).to(device)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        h = self.tok_embeddings(input_ids)
        cosin_m_thetas = self.cosin_m_thetas[positions]
        mask = None
        if input_ids.shape[1] > 1:
            # causal mask, prefill only
            seqlen = input_ids.shape[1]
            t = torch.full((seqlen, seqlen), dtype=h.dtype, fill_value=1, device=h.device)
            mask = torch.tril(t, diagonal=0).to(h.dtype)
            mask = torch.triu(mask, diagonal=-(self.args.sliding_window - 1))  # drop anything older than W
            mask = torch.log(mask)

        for layer in self.layers:
            h = layer(h, cosin_m_thetas, positions, mask)
        return self.output(self.norm(h)).float()

    def allocate_cache(self, batch_size: int, max_seq_len: int) -> None:
        """Give every layer a zeroed KV cache for exactly this batch shape, in the weights' dtype."""
        weight = self.output.weight
        shape = (batch_size, max_seq_len, self.args.n_kv_heads, self.args.head_dim)
        for layer in self.layers:
            layer.attention.cache_k = torch.zeros(shape, dtype=weight.dtype, device=weight.device)
            layer.attention.cache_v = torch.zeros(shape, dtype=weight.dtype, device=weight.device)

    def release_cache(self) -> None:
        for layer in self.layers:
            layer.attention.cache_k = layer.attention.cache_v = None


@torch.no_grad()
def generate(prompts: List[str], model: Transformer, tokenizer: "Tokenizer", max_tokens: int, device=None):
    """Greedy text completion, as the notebook's demo() runs it."""
    device = device or next(model.parameters()).device
    encoded_prompts = [tokenizer.encode(p, bos=True, eos=False) for p in prompts]
    min_prompt_len = min(len(x) for x in encoded_prompts)
    max_prompt_len = max(len(x) for x in encoded_prompts)
    model.allocate_cache(len(prompts), max_prompt_len + max_tokens)

    pad_id = tokenizer._model.pad_id()
    input_tokens = torch.full((len(prompts), max_prompt_len), pad_id, dtype=torch.long, device=device)
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != pad_id

    # prefill as far as the shortest prompt, so every row sits at the same position
    positions = torch.arange(0, min_prompt_len, device=device)
    logits = model.forward(input_tokens[:, :min_prompt_len], positions)
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    # greedy decode through the kv cache
    generated, cur_pos = [], min_prompt_len
    for _ in range(max_tokens):
        next_token = torch.argmax(logprobs[:, -1, :], dim=-1)
        if cur_pos < input_mask.shape[1]:
            # longer prompts are still being read: feed their next prompt token, not the prediction
            next_token = torch.where(input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token)
        generated.append(next_token[:, None])
        logits = model.forward(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        cur_pos += 1

    res = []
    if max_tokens > 0:
        generated = torch.cat(generated, 1)
        for i, x in enumerate(encoded_prompts):
            res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))
    return res
