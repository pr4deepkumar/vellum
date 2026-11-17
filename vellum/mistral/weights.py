"""Mistral's consolidated checkpoint (consolidated.safetensors + params.json) and its tokenizer.

`source` is a local folder holding the files (e.g. the notebook's WEIGHTS_DIR, or a folder
written by `vellum quantize`) or a Hugging Face repo id such as mistralai/Mistral-7B-Instruct-v0.3,
from which only the needed files download.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .awq import describe_plan, make_quantized, tune
from .model import ROPE_POSITIONS, Transformer, TransformerArgs, precompute_thetas

if TYPE_CHECKING:
    from mistral_common.tokens.tokenizers.base import Tokenizer

WEIGHT_FILES = ("consolidated.safetensors", "params.json")
QUANTIZATION_FILE = "quantization.json"  # present only in folders written by save_quantized
TOKENIZER_FILE = "tokenizer.model.v3"


def _folder(source: str | Path, files) -> Path:
    folder = Path(source)
    if folder.is_dir():
        return folder
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(source), allow_patterns=list(files)))


def load_model(source: str | Path, device="cuda", dtype=torch.float16, triton: bool = True,
               int4pack: bool = True) -> Transformer:
    """The model with real weights, without a detour through host RAM.

    Modules are built on the meta device and each tensor streams from the file straight to
    `device` in `dtype`: no random fp32 init (~29 GB of host RAM for 7B) and no second copy of
    the checkpoint. A quantized checkpoint loads in bf16 and, on a GPU, times its matmul paths
    per layer shape and row count: awq.tune(triton=..., int4pack=...) says which are allowed.
    KV caches stay unallocated until Transformer.allocate_cache.
    """
    from safetensors import safe_open

    folder = _folder(source, WEIGHT_FILES)
    args = TransformerArgs.from_params(json.loads((folder / "params.json").read_text()))
    quantized = (folder / QUANTIZATION_FILE).exists()
    if quantized:
        dtype = torch.bfloat16
    with torch.device("meta"):
        model = Transformer(args, device="meta")
        if quantized:
            make_quantized(model)
    with safe_open(str(folder / "consolidated.safetensors"), framework="pt", device=str(device)) as f:
        state = {}
        for name in f.keys():
            t = f.get_tensor(name)
            state[name] = t.to(dtype) if t.is_floating_point() and not name.endswith(".sz") else t
    model.load_state_dict(state, strict=True, assign=True)
    model.cosin_m_thetas = precompute_thetas(args.head_dim, ROPE_POSITIONS, args.rope_N).to(device)
    if quantized:
        for (o, i), plan in sorted(tune(model, triton=triton, int4pack=int4pack).items()):
            print(f"  int4 {o}x{i}: {describe_plan(plan)}", flush=True)
    return model.eval()


def save_quantized(model: Transformer, folder: str | Path, details: dict) -> Path:
    """Write a model from awq_quantize as a checkpoint load_model reads back (about 4 GB for 7B)."""
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(folder / "consolidated.safetensors"))
    (folder / "params.json").write_text(json.dumps(dataclasses.asdict(model.args), indent=2))
    (folder / QUANTIZATION_FILE).write_text(json.dumps(details, indent=2))
    return folder


def load_tokenizer(source: str | Path) -> "Tokenizer":
    """Mistral's v3 SentencePiece tokenizer, as the notebook uses it (needs mistral_common)."""
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    folder = _folder(source, (TOKENIZER_FILE,))
    return MistralTokenizer.from_file(str(folder / TOKENIZER_FILE)).instruct_tokenizer.tokenizer
