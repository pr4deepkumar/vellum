"""The from-scratch model as a benchmarkable engine, optimized one measured rung at a time.

`model.py` is what the notebook builds. This module drives those same modules and weights
through a ladder of serving optimizations, so `vellum offline --backend scratch --rung <rung>`
can pin every change in latency, throughput, memory or peak batch size on the one change that
caused it. Each rung keeps everything below it:

    naive      the notebook model untouched: logits for every prompt position, attention via
               repeat_kv + matmul, and a host sync per layer per decode step (.item())
    lastlogit  prefill projects only the last position onto the vocab, instead of
               materialising (batch, prompt_len, 32768) fp32 logits to keep one row
    nosync     the decode position stays a Python int, so nothing waits on the GPU mid-step
    gqa        decode lets the query heads that share a KV head read it in place, instead of
               repeat_kv copying the whole cache n_heads / n_kv_heads times every step
    sdpa       fused scaled_dot_product_attention kernels (FlashAttention on prefill)
    chunked    prefill runs in chunks of at most `prefill_tokens` tokens across the batch, each
               attending to the cache plus itself (the paper's pre-fill and chunking), so a big
               batch's prompt activations no longer have to fit at once
    graph      one decode step is captured as a CUDA graph and replayed (CUDA only)
    awq        the model's linears hold 4-bit AWQ weights (awq.py), multiplied by PyTorch's int4
               kernel or, for prefill, a rebuilt dense weight; the code path is graph's
    triton     VELLUM's own W4A16 Triton kernels (w4a16.py) join those paths, used wherever they
               measure faster; the model must be loaded with load_model(..., triton=True)

Every rung up to graph computes the same function as `naive`, and tests/test_mistral_engine.py
holds them to it. awq changes the weights, so it is held to a perplexity budget instead; triton
computes awq's function with other kernels.

The optimized rungs also apply the sliding window during decode, which the notebook masks only on
prefill; neither shows below `sliding_window` tokens. v0.3 itself was trained without a window
(its HF config has `sliding_window: null`), so past 4096 tokens this model and HF / vLLM attend
differently.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .attention import allowed, notebook_attention, notebook_mask, sdpa_chunk, sdpa_prefill
from .awq import QuantLinear, tuned_with_triton
from .model import Transformer, apply_rotary_emb, repeat_kv

LADDER = ("naive", "lastlogit", "nosync", "gqa", "sdpa", "chunked", "graph", "awq", "triton")
PREFILL_TOKENS = 8192  # chunked rung: prompt tokens per forward, summed over the batch


class Engine:
    """Greedy, fixed-length generation for a static batch, at one rung of LADDER.

        engine = Engine(model, rung="gqa")
        engine.setup(batch_size=8, max_seq_len=640)   # KV cache for exactly this shape
        tokens = engine.generate(prompt_ids, max_new_tokens=128)

    The cache lives on the model's attention modules, where the notebook put it, so drive
    one Engine per model at a time.
    """

    def __init__(self, model: Transformer, rung: str = "graph", prefill_tokens: int = PREFILL_TOKENS):
        if rung not in LADDER:
            raise ValueError(f"unknown rung {rung!r}; pick one of {', '.join(LADDER)}")
        on = set(LADDER[1 : LADDER.index(rung) + 1])
        if "awq" in on and not any(isinstance(m, QuantLinear) for m in model.modules()):
            raise ValueError("rung 'awq' needs a 4-bit model from `vellum quantize`")
        param = next(model.parameters())
        if "graph" in on and param.device.type != "cuda":
            raise ValueError(f"rung {rung!r} captures a CUDA graph and needs a CUDA device")
        if "triton" in on and not tuned_with_triton():
            raise ValueError("rung 'triton' needs the model loaded with load_model(..., triton=True)")
        self.model, self.rung, self.on = model, rung, on
        self.prefill_tokens = prefill_tokens
        self.device, self.dtype = param.device, param.dtype
        self.batch_size = self.max_seq_len = 0
        self._graph = self._static = None

    # -- cache -------------------------------------------------------------------------

    def setup(self, batch_size: int, max_seq_len: int) -> None:
        """Allocate the KV cache for exactly this batch shape, and capture the decode graph."""
        self.teardown()
        # zeroed: the graph step attends over the whole cache under a mask, and a masked slot
        # still multiplies its V by 0 -- uninitialised NaNs would leak through
        self.model.allocate_cache(batch_size, max_seq_len)
        self.batch_size, self.max_seq_len = batch_size, max_seq_len
        self._slots = torch.arange(max_seq_len, device=self.device)
        if "graph" in self.on:
            self._capture()

    def teardown(self) -> None:
        """Drop the cache and graph so the next batch shape can have the memory."""
        self._graph = self._static = None
        self.model.release_cache()
        self.batch_size = self.max_seq_len = 0

    # -- generation --------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Prefill `ids` (batch, prompt_len), then greedy-decode exactly `max_new_tokens` per
        row -- no EOS stop, like VELLUM's ignore_eos. Returns (batch, max_new_tokens)."""
        b, n = ids.shape
        if b != self.batch_size or n + max_new_tokens > self.max_seq_len:
            raise ValueError(f"{b} x ({n} + {max_new_tokens}) tokens does not fit the cache set up "
                             f"for {self.batch_size} x {self.max_seq_len}; call setup() first")
        out = [self.prefill(ids).argmax(-1)]
        for i in range(max_new_tokens - 1):
            out.append(self.decode(out[-1], n + i).argmax(-1))
        return torch.stack(out, 1)

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        """Run a prompt (batch, n) from position 0; logits (batch, vocab) for its last position."""
        if "chunked" in self.on:
            step = max(1, self.prefill_tokens // ids.shape[0])
            for start in range(0, ids.shape[1], step):
                logits = self._forward(ids[:, start : start + step], start=start)
            return logits
        if "nosync" in self.on:
            return self._forward(ids, start=0)
        positions = torch.arange(ids.shape[1], device=self.device)
        if "lastlogit" in self.on:
            return self._notebook_forward_last(ids, positions)
        return self.model(ids, positions)[:, -1]  # the notebook's forward, verbatim

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor, pos: int) -> torch.Tensor:
        """Feed one token per row (batch,) at position `pos`; logits (batch, vocab)."""
        if "graph" in self.on:
            return self._replay(tokens, pos)
        if "nosync" in self.on:
            return self._forward(tokens[:, None], start=pos)
        # as in the notebook's generate(): the position goes host -> device every step, and
        # every attention layer reads it back with .item()
        return self.model(tokens[:, None], torch.tensor([pos], device=self.device))[:, -1]

    # -- rungs ---------------------------------------------------------------------------

    def _notebook_forward_last(self, ids, positions):
        """The notebook's Transformer.forward, projecting only the last position to the vocab."""
        m = self.model
        h = m.tok_embeddings(ids)
        cs = m.cosin_m_thetas[positions]
        mask = notebook_mask(ids.shape[1], m.args.sliding_window, h) if ids.shape[1] > 1 else None
        for layer in m.layers:
            h = layer(h, cs, positions, mask)
        return m.output(m.norm(h[:, -1])).float()

    def _forward(self, ids, start: int):
        """Eager forward from the nosync rung up (and the prefill of the rungs above)."""
        m = self.model
        n = ids.shape[1]
        h = m.tok_embeddings(ids)
        cs = m.cosin_m_thetas[start : start + n]
        mask = None
        if n > 1 and "sdpa" not in self.on:
            mask = notebook_mask(n, m.args.sliding_window, h)
        for layer in m.layers:
            h = h + self._attention(layer.attention, layer.attention_norm(h), cs, start, mask)
            h = h + layer.feed_forward(layer.ffn_norm(h))
        return m.output(m.norm(h[:, -1])).float()

    def _attention(self, attn, x, cs, start: int, mask):
        b, n, _ = x.shape
        hd, nkv, rep, window = attn.args.head_dim, attn.n_kv_heads, attn.repeats, attn.args.sliding_window
        q = apply_rotary_emb(attn.wq(x).view(b, n, attn.n_heads, hd), cs)
        k = apply_rotary_emb(attn.wk(x).view(b, n, nkv, hd), cs)
        v = attn.wv(x).view(b, n, nkv, hd)
        attn.cache_k[:, start : start + n] = k  # plain slice write: start is a Python int
        attn.cache_v[:, start : start + n] = v

        if n > 1 and start > 0:  # a later prefill chunk: the context is the cache through this chunk
            lo = max(0, start - window + 1)
            k, v = repeat_kv(attn.cache_k[:, lo : start + n], attn.cache_v[:, lo : start + n], rep)
            return attn.wo(sdpa_chunk(q, k, v, start, lo, window))
        if n > 1:  # prefill from position 0: the fresh k/v are the whole context
            k, v = repeat_kv(k, v, rep)  # prefill is compute-bound; this copy is small next to the matmuls
            if "sdpa" in self.on:
                return attn.wo(sdpa_prefill(q, k, v, window))
            return attn.wo(notebook_attention(q, k, v, mask, attn.scale))

        lo = max(0, start + 1 - window)
        k, v = attn.cache_k[:, lo : start + 1], attn.cache_v[:, lo : start + 1]
        if "gqa" not in self.on:
            k, v = repeat_kv(k, v, rep)  # the notebook's decode: a fresh rep-times copy of the cache
            return attn.wo(notebook_attention(q, k, v, None, attn.scale))
        # fold the `rep` query heads that share each KV head into the query axis, so K/V are
        # read where they sit. Head h uses KV head h // rep, the pairing repeat_kv produces.
        q = q.view(b, nkv, rep, hd)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if "sdpa" in self.on:
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            o = F.softmax(q @ k.transpose(2, 3) * attn.scale, dim=-1).type_as(q) @ v
        return attn.wo(o.reshape(b, 1, -1))

    # -- graph rung ----------------------------------------------------------------------

    def _decode_static(self, tokens, pos):
        """One decode step with fixed shapes and no host round-trips, so it can be captured.

        tokens (batch, 1) and pos (1,) are device tensors. Attention spans the whole cache
        under a window mask instead of slicing [:pos + 1], which would change shape every step.
        """
        m = self.model
        b = tokens.shape[0]
        h = m.tok_embeddings(tokens)
        cs = m.cosin_m_thetas[pos]
        mask = allowed(pos, self._slots, m.args.sliding_window).view(1, 1, 1, -1)
        for layer in m.layers:
            a = layer.attention
            hd, nkv = a.args.head_dim, a.n_kv_heads
            x = layer.attention_norm(h)
            q = apply_rotary_emb(a.wq(x).view(b, 1, a.n_heads, hd), cs)
            k = apply_rotary_emb(a.wk(x).view(b, 1, nkv, hd), cs)
            a.cache_k.index_copy_(1, pos, k)
            a.cache_v.index_copy_(1, pos, a.wv(x).view(b, 1, nkv, hd))
            o = F.scaled_dot_product_attention(q.view(b, nkv, a.repeats, hd), a.cache_k.transpose(1, 2),
                                               a.cache_v.transpose(1, 2), attn_mask=mask)
            h = h + a.wo(o.reshape(b, 1, -1))
            h = h + layer.feed_forward(layer.ffn_norm(h))
        return m.output(m.norm(h[:, -1])).float()

    def _capture(self) -> None:
        """Record one decode step as a CUDA graph; decode() then refreshes inputs and replays."""
        tokens = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
        pos = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):  # warm up cuBLAS handles and the allocator off-capture
                for _ in range(3):
                    self._decode_static(tokens, pos)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits = self._decode_static(tokens, pos)
            graph.replay()  # a replay must reproduce the eager step it recorded
            if not torch.allclose(logits, self._decode_static(tokens, pos), rtol=1e-3, atol=1e-3):
                raise RuntimeError("CUDA graph replay diverged from the eager decode step")
        # warm-up and capture wrote junk K/V at slot 0; prefill overwrites it before decode reads it
        self._graph, self._static = graph, (tokens, pos, logits)

    def _replay(self, tokens, pos: int):
        static_tokens, static_pos, logits = self._static
        static_tokens.copy_(tokens[:, None])
        static_pos.fill_(pos)
        self._graph.replay()
        return logits.clone()  # the next replay overwrites the static buffer
