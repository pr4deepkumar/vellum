"""The optimization ladder must not change what the model computes.

CPU, fp32, a tiny random config, no downloads. Every rung is held to the notebook model's
logits; the graph rung's decode step is checked eagerly (capturing it needs CUDA); and
load_model must rebuild the same model from a consolidated.safetensors checkpoint.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from vellum.mistral import LADDER, Engine, Transformer, TransformerArgs, load_model  # noqa: E402

TINY = dict(dim=64, n_layers=2, head_dim=16, hidden_dim=128, n_heads=4, n_kv_heads=2, vocab_size=100)
EAGER_RUNGS = LADDER[: LADDER.index("graph")]  # graph (and awq above it) capture CUDA graphs


def tiny_model(**overrides) -> Transformer:
    torch.manual_seed(0)
    return Transformer(TransformerArgs(**{**TINY, **overrides})).eval()


def randint(*shape, seed=1):
    return torch.randint(0, TINY["vocab_size"], shape, generator=torch.Generator().manual_seed(seed))


def trace(engine, prompt, fed):
    """Logits after the prompt, then after each token of `fed` decoded in turn: (batch, 1 + len, vocab).

    Every engine is fed the same tokens, so one near-tie argmax can't derail a comparison.
    """
    b, n = prompt.shape
    engine.setup(b, n + fed.shape[1])
    out = [engine.prefill(prompt)]
    for i in range(fed.shape[1]):
        out.append(engine.decode(fed[:, i], n + i))
    return torch.stack(out, 1)


@pytest.mark.parametrize("rung", EAGER_RUNGS)
def test_every_rung_computes_the_notebook_model(rung):
    model = tiny_model()
    prompt, fed = randint(3, 9), randint(3, 6, seed=2)
    want = trace(Engine(model, "naive"), prompt, fed)
    torch.testing.assert_close(trace(Engine(model, rung), prompt, fed), want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rung", ["nosync", "gqa", "sdpa"])
def test_decode_past_the_window_matches_full_recompute(rung):
    """Beyond W tokens a decoded token must see exactly what the prefill mask would let it see."""
    model = tiny_model(sliding_window=4)
    seq = randint(2, 11)
    engine = Engine(model, rung)
    logits = trace(engine, seq[:, :5], seq[:, 5:])  # 5-token prefill already crosses W
    full = model(seq, torch.arange(seq.shape[1]))[:, -1]  # notebook prefill: window via its mask
    torch.testing.assert_close(logits[:, -1], full, rtol=1e-5, atol=1e-5)


def test_chunked_prefill_matches_one_pass():
    """Prefill in chunks of 2 tokens, each attending the cache plus itself, == the notebook's one pass."""
    model = tiny_model()
    prompt, fed = randint(3, 9), randint(3, 4, seed=2)
    want = trace(Engine(model, "naive"), prompt, fed)
    chunked = Engine(model, "chunked", prefill_tokens=6)  # 6 tokens over a batch of 3 -> 2 per chunk
    torch.testing.assert_close(trace(chunked, prompt, fed), want, rtol=1e-5, atol=1e-5)


def test_chunked_prefill_past_the_window_matches_full_recompute():
    """Chunks that start past W must drop keys older than the window, like the prefill mask."""
    model = tiny_model(sliding_window=4)
    seq = randint(2, 11)
    engine = Engine(model, "chunked", prefill_tokens=6)  # 3 tokens per chunk: starts 0, 3, 6, 9
    logits = trace(engine, seq, seq[:, :0])
    full = model(seq, torch.arange(seq.shape[1]))[:, -1]
    torch.testing.assert_close(logits[:, -1], full, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_graph_decode_step_matches_eager_decode():
    """The step the graph rung captures reads the whole cache under a mask instead of slicing it.
    Run eagerly, it must agree with the sliced decode, window included."""
    model = tiny_model(sliding_window=4)
    prompt, fed = randint(2, 6), randint(2, 5, seed=2)
    want = trace(Engine(model, "sdpa"), prompt, fed)
    engine = Engine(model, "sdpa")
    engine.setup(2, 6 + 5)
    got = [engine.prefill(prompt)]
    for i in range(fed.shape[1]):
        got.append(engine._decode_static(fed[:, i : i + 1], torch.tensor([6 + i])))
    torch.testing.assert_close(torch.stack(got, 1), want, rtol=1e-5, atol=1e-5)


def test_generate_feeds_back_its_own_argmax():
    model = tiny_model()
    prompt = randint(2, 5)
    engine = Engine(model, "sdpa")
    engine.setup(2, 5 + 6)
    tokens = engine.generate(prompt, 6)
    assert tokens.shape == (2, 6)
    logits = trace(Engine(model, "sdpa"), prompt, tokens[:, :-1])
    assert torch.equal(logits.argmax(-1), tokens)


def test_generate_checks_the_cache_shape():
    engine = Engine(tiny_model(), "sdpa")
    engine.setup(2, 8)
    with pytest.raises(ValueError, match="setup"):
        engine.generate(randint(2, 6), 4)  # 6 + 4 positions > 8


def test_unknown_rung_is_rejected():
    with pytest.raises(ValueError, match="naive"):
        Engine(tiny_model(), "turbo")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU error path")
def test_graph_rung_needs_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        Engine(tiny_model(), "graph")


@pytest.mark.parametrize("rung", ["awq", "triton"])
def test_4bit_rungs_need_a_quantized_model(rung):
    with pytest.raises(ValueError, match="vellum quantize"):
        Engine(tiny_model(), rung)


def test_load_model_rebuilds_the_checkpoint(tmp_path):
    """Meta-device build + tensors streamed from the file == the model that was saved,
    with RoPE's base read from params.json's rope_theta."""
    from safetensors.torch import save_file

    model = tiny_model(rope_N=10000.0)
    save_file(model.state_dict(), str(tmp_path / "consolidated.safetensors"))
    params = {**TINY, "norm_eps": 1e-5, "rope_theta": 10000.0}  # the keys Mistral ships
    (tmp_path / "params.json").write_text(json.dumps(params))

    loaded = load_model(tmp_path, device="cpu", dtype=torch.float32)
    assert loaded.args.rope_N == 10000.0
    prompt, fed = randint(2, 7), randint(2, 3, seed=2)
    torch.testing.assert_close(trace(Engine(loaded, "sdpa"), prompt, fed),
                               trace(Engine(model, "sdpa"), prompt, fed))

