"""notebooks/01_build_mistral.ipynb builds exactly the model vellum.mistral benchmarks.

Runs the notebook's model-defining cells (tagged "model") on CPU, loads one random tiny
checkpoint into both models, and holds them to identical outputs. That is what lets the
`naive` rung of the ladder stand for "the model the notebook builds".
"""
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")  # the notebook imports it

from vellum.mistral import Transformer, TransformerArgs, generate  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_build_mistral.ipynb"
TINY = dict(dim=64, n_layers=2, head_dim=16, hidden_dim=128, n_heads=4, n_kv_heads=2, vocab_size=100,
            max_seq_len=64, max_batch_size=2)


@pytest.fixture
def notebook(monkeypatch):
    """The notebook's namespace after its "model" cells; its tokenizer and config imports stubbed."""
    stubs = {
        "mistral_common": {}, "mistral_common.tokens": {}, "mistral_common.tokens.tokenizers": {},
        "mistral_common.tokens.tokenizers.base": {"Tokenizer": object},
        "mistral_common.tokens.tokenizers.mistral": {"MistralTokenizer": object},
        "simple_parsing": {}, "simple_parsing.helpers": {"Serializable": object},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    ns = {"DEVICE": "cpu"}  # the notebook sets this in its setup cell
    cells = [c for c in json.loads(NOTEBOOK.read_text())["cells"]
             if "model" in c.get("metadata", {}).get("tags", [])]
    assert len(cells) >= 8, "the notebook's model cells lost their 'model' tag"
    for cell in cells:
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), ns)
    return ns


class _Tokenizer:
    class _model:  # noqa: N801 -- mirrors tokenizer._model
        @staticmethod
        def pad_id():
            return -1

    def encode(self, s, bos, eos):
        return [1] + [int(t) for t in s.split()]

    def decode(self, ids):
        return ids


@torch.no_grad()
def test_notebook_builds_the_package_model(notebook):
    torch.manual_seed(0)
    built = notebook["Transformer"](notebook["TransformerArgs"](**TINY)).eval()
    package = Transformer(TransformerArgs(**TINY)).eval()
    package.load_state_dict(built.state_dict(), strict=True)  # same parameter names, same shapes

    prompt = torch.tensor([[1, 5, 17, 42, 9, 3, 77]])
    n = prompt.shape[1]
    assert torch.equal(built(prompt, torch.arange(n)), package(prompt, torch.arange(n)))  # prefill
    token = torch.tensor([[12]])
    assert torch.equal(built(token, torch.tensor([n])), package(token, torch.tensor([n])))  # decode

    prompts = ["5 17 42 9 33", "3 77"]
    assert notebook["generate"](prompts, built, _Tokenizer(), 4) == generate(prompts, package, _Tokenizer(), 4)
