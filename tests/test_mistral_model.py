"""The from-scratch Mistral itself (vellum.mistral.model): CPU, tiny random config, no weights."""
import pytest

torch = pytest.importorskip("torch")

from vellum.mistral import Transformer, TransformerArgs, generate, precompute_thetas  # noqa: E402

TINY = dict(dim=64, n_layers=2, head_dim=16, hidden_dim=128, n_heads=4, n_kv_heads=2, vocab_size=100,
            max_seq_len=64)


def tiny_model(**overrides) -> Transformer:
    torch.manual_seed(0)
    return Transformer(TransformerArgs(**{**TINY, **overrides})).eval()


def test_rope_leaves_position_zero_unrotated():
    cs = precompute_thetas(8, 2, 1e6)
    assert cs.shape == (2, 4, 2)  # positions x head_dim/2 x (cos, sin)
    assert torch.equal(cs[0, :, 0], torch.ones(4)) and torch.equal(cs[0, :, 1], torch.zeros(4))


@torch.no_grad()
def test_cached_decode_matches_full_recompute():
    model = tiny_model()
    prompt = torch.tensor([[1, 5, 17, 42, 9, 3, 77]])
    n = prompt.shape[1]
    logits = model(prompt, torch.arange(n))
    assert logits.shape == (1, n, TINY["vocab_size"]) and torch.isfinite(logits).all()
    tokens = []
    for i in range(3):  # greedy decode through the KV cache
        tokens.append(logits[:, -1].argmax(-1))
        logits = model(tokens[-1][:, None], torch.tensor([n + i]))
    full = torch.cat([prompt, torch.stack(tokens, 1)], 1)
    fresh = tiny_model()
    fresh.load_state_dict(model.state_dict())
    torch.testing.assert_close(fresh(full, torch.arange(full.shape[1]))[:, -1], logits[:, -1],
                               rtol=1e-4, atol=1e-4)


def test_args_read_params_json():
    args = TransformerArgs.from_params({"dim": 64, "rope_theta": 1e4, "some_future_key": 1})
    assert args.dim == 64 and args.rope_N == 1e4


class Tokenizer:
    """Just enough of mistral_common's tokenizer for generate(): '5 17 42' <-> [1, 5, 17, 42]."""

    class _model:  # noqa: N801 -- mirrors tokenizer._model
        @staticmethod
        def pad_id():
            return -1

    def encode(self, s, bos, eos):
        return [1] + [int(t) for t in s.split()]

    def decode(self, ids):
        return ids


def test_batched_generate_keeps_every_prompt():
    """A prompt longer than the shortest in its batch is read in full, not truncated to it."""
    model = tiny_model()
    long, short = "5 17 42 9 33", "3 77"
    alone = generate([long], model, Tokenizer(), max_tokens=4)[0]
    batched = generate([long, short], model, Tokenizer(), max_tokens=4)[0]
    assert batched[:6] == [1, 5, 17, 42, 9, 33]
    assert alone[: len(batched)] == batched
