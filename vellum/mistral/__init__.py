"""Mistral 7B built from scratch in plain PyTorch, and the optimization ladder VELLUM benchmarks.

    model      the model notebooks/01_build_mistral.ipynb builds: the ladder's `naive` rung
    engine     that model driven at one rung of LADDER, naive -> lastlogit -> ... -> triton
    awq        4-bit activation-aware weight quantization, and a perplexity check for it
    w4a16      the Triton kernels that multiply by those 4-bit weights on a GPU
    attention  the attention math the rungs share
    weights    Mistral's consolidated checkpoint and tokenizer, from a folder or the Hub

Needs torch (`pip install -e ".[mistral]"`); the rest of VELLUM does not.
"""

from .awq import QuantLinear, awq_quantize, perplexity
from .engine import LADDER, Engine
from .model import (
    Attention,
    FeedForward,
    RMSNorm,
    Transformer,
    TransformerArgs,
    TransformerBlock,
    apply_rotary_emb,
    generate,
    precompute_thetas,
    repeat_kv,
)
from .weights import load_model, load_tokenizer, save_quantized

__all__ = [
    "LADDER", "Engine",
    "TransformerArgs", "Transformer", "TransformerBlock", "Attention", "FeedForward", "RMSNorm",
    "precompute_thetas", "apply_rotary_emb", "repeat_kv", "generate",
    "QuantLinear", "awq_quantize", "perplexity",
    "load_model", "load_tokenizer", "save_quantized",
]
