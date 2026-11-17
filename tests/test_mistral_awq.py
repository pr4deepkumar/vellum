"""4-bit AWQ (vellum.mistral.awq) on CPU, on a small random model whose widths are multiples of 128.

Quantized weights cannot reproduce the full-precision model exactly, so these hold each piece to
its own contract: rounding error is bounded, folding scales is exact, the scale search never does
worse than plain round-to-nearest, every path reads the codes as the reference does, and
checkpoints round-trip. The Triton kernels run on a GPU when there is one, and in Triton's
interpreter on CPU wherever Triton is installed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

import vellum.mistral.awq as awq_module  # noqa: E402
from vellum.mistral import (  # noqa: E402
    Engine, QuantLinear, Transformer, TransformerArgs, awq_quantize, load_model, perplexity, save_quantized,
    w4a16,
)
from vellum.mistral.awq import (  # noqa: E402
    GROUP, _fold, _output_error, dequantize, pack_nibbles, quantize, route, search_scale, unpack_nibbles,
)

needs_triton_gpu = pytest.mark.skipif(not w4a16.available(), reason="the Triton kernels need CUDA and Triton")

SMALL = dict(dim=128, n_layers=2, head_dim=32, hidden_dim=256, n_heads=4, n_kv_heads=2, vocab_size=100)


def small_model(**overrides) -> Transformer:
    torch.manual_seed(0)
    return Transformer(TransformerArgs(**{**SMALL, **overrides})).eval()


def randint(*shape, seed=1):
    return torch.randint(0, SMALL["vocab_size"], shape, generator=torch.Generator().manual_seed(seed))


def outlier_inputs(rows=512, width=256, seed=0):
    """Activations where a few channels run 30x hotter, the pattern AWQ exists for."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, width, generator=g)
    x[:, :4] *= 30
    return x


def test_rounding_error_is_at_most_half_a_step():
    w = torch.randn(64, 2 * GROUP) * 0.05
    q, sz = quantize(w)
    assert q.dtype == torch.int32 and int(q.min()) >= 0 and int(q.max()) <= 15
    assert sz.shape == (2, 64, 2) and sz.dtype == torch.bfloat16
    step = sz[..., 0].t().float().repeat_interleave(GROUP, 1)
    assert ((dequantize(q, sz) - w).abs() <= step / 2 * 1.02 + 1e-6).all()


def test_nibbles_round_trip():
    q = torch.randint(0, 16, (8, 256), dtype=torch.int32)
    packed = pack_nibbles(q)
    assert packed.dtype == torch.uint8 and packed.shape == (8, 128)
    assert torch.equal(unpack_nibbles(packed), q)
    # column 2i in the high nibble, 2i + 1 in the low one: the layout the Triton kernels read
    assert pack_nibbles(torch.tensor([[0x1, 0x2, 0xA, 0xF]], dtype=torch.int32)).tolist() == [[0x12, 0xAF]]


def rel_err(got, want):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


@torch.no_grad()
def test_untuned_layers_take_the_dense_path():
    layer = QuantLinear.from_weight(torch.randn(256, 384) * 0.05)
    x = torch.randn(3, 5, 384, dtype=torch.bfloat16)
    want = torch.nn.functional.linear(x, layer.dense_weight(torch.bfloat16))
    assert torch.equal(layer(x), want)  # on CPU, and on CUDA before tune(), there is no plan to follow
    assert awq_module.tune(torch.nn.ModuleList([layer])) == {}  # nothing to time off the GPU


def test_a_plan_routes_each_row_count_to_its_kernel():
    a, b = dict(block_m=1), dict(block_m=16)
    plan = ((2, "gemv", a), (64, "gemm", b), (8192, "dense", {}))
    assert [route(plan, r) for r in (1, 2, 3, 64, 65, 8192, 10_000)] == [
        ("gemv", a), ("gemv", a), ("gemm", b), ("gemm", b), ("dense", {}), ("dense", {}), ("dense", {})]
    gemv = dict(block_m=1, block_n=32, groups=1, num_warps=4, num_stages=1)
    assert awq_module.describe_plan(((2, "gemv", gemv), (8192, "dense", {}))) == (
        "gemv 1x32x1g w4 to 2 rows, dense from 3 rows")


# every gemv tile at one warp count, two more warp counts, and every gemm_t and gemm tile
KERNEL_CASES = [("gemv", c) for c in w4a16.GEMV_CONFIGS if c["num_warps"] == 4]
KERNEL_CASES += [("gemv", dict(block_m=m, block_n=32, groups=2, num_warps=w, num_stages=1)) for m, w in ((1, 2), (4, 8))]
KERNEL_CASES += [("gemm_t", c) for c in w4a16.GEMM_T_CONFIGS] + [("gemm", c) for c in w4a16.GEMM_CONFIGS]
ROWS_TO_CHECK = {"gemv": (1, 3, 9), "gemm_t": (1, 5, 16, 33), "gemm": (1, 5, 16, 33, 130)}


@needs_triton_gpu
@pytest.mark.parametrize("kernel,cfg", KERNEL_CASES, ids=[w4a16.describe(k, c) for k, c in KERNEL_CASES])
@torch.no_grad()
def test_triton_kernels_match_the_dequantized_weight(kernel, cfg):
    """200 outputs (not a tile multiple), 4 groups, row counts that leave partial tiles, and an x
    whose rows are not contiguous with each other."""
    from triton.runtime.errors import OutOfResources

    torch.manual_seed(0)
    layer = QuantLinear.from_weight(torch.randn(200, 512, device="cuda") * 0.05)
    reference = layer.dense_weight(torch.float32)
    for rows in ROWS_TO_CHECK[kernel]:
        x = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)[:, :512]
        try:
            got = w4a16.matmul(kernel, cfg, x, layer.nibbles, layer.sz)
        except OutOfResources:
            pytest.skip("this tile needs more shared memory than this GPU has")
        assert got.shape == (rows, 200) and got.dtype == torch.bfloat16
        assert rel_err(got, x.float() @ reference.t()) < 1e-2, rows


@needs_triton_gpu
@torch.no_grad()
def test_forward_follows_the_plan_and_replays_in_a_cuda_graph(monkeypatch):
    layer = QuantLinear.from_weight(torch.randn(256, 512, device="cuda") * 0.05)
    layer._int4pack = awq_module.int4pack_layout(layer.nibbles)
    gemv = dict(block_m=1, block_n=32, groups=1, num_warps=4, num_stages=1)
    gemm_t = dict(block_m=16, block_n=64, block_kh=64, num_warps=4, num_stages=3)
    gemm = dict(block_m=16, block_n=64, block_kh=64, num_warps=4, num_stages=3)
    monkeypatch.setattr(awq_module, "_PLANS", {(256, 512): ((1, "gemv", gemv), (4, "int4pack", {}),
                                                            (8, "gemm_t", gemm_t), (64, "gemm", gemm),
                                                            (8192, "dense", {}))})
    calls = []
    real, real_int4pack = w4a16.matmul, awq_module.int4pack_mm
    monkeypatch.setattr(w4a16, "matmul", lambda kernel, *a: calls.append(kernel) or real(kernel, *a))
    monkeypatch.setattr(awq_module, "int4pack_mm", lambda *a: calls.append("int4pack") or real_int4pack(*a))
    reference = layer.dense_weight(torch.float32)
    for rows, path in ((1, ["gemv"]), (3, ["int4pack"]), (8, ["gemm_t"]), (40, ["gemm"]), (100, [])):
        x = torch.randn(2, rows, 512, device="cuda", dtype=torch.bfloat16)[0]  # (rows, 512), from a batch
        calls.clear()
        eager = layer(x)
        assert calls == path, rows
        assert rel_err(eager, x.float() @ reference.t()) < 1e-2
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):  # the graph rung captures decode, kernels and all
            replayed = layer(x)
        graph.replay()
        if path and path != ["int4pack"]:  # the Triton kernels use no atomics: bit for bit
            assert torch.equal(replayed, eager), rows
        else:  # PyTorch's kernels may pick another algorithm under capture
            assert rel_err(replayed, eager) < 1e-3, rows
    layer._int4pack = None  # single-copy: a plan's int4pack entry falls back to the dense path
    calls.clear()
    x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    assert rel_err(layer(x), x.float() @ reference.t()) < 1e-2 and calls == []


@needs_triton_gpu
@pytest.mark.parametrize("triton,int4pack", [(True, True), (True, False), (False, True)],
                         ids=["hybrid (rung triton)", "single-copy", "stock (rung awq)"])
@torch.no_grad()
def test_tune_picks_a_checked_path_per_row_count(monkeypatch, triton, int4pack):
    monkeypatch.setattr(awq_module, "_PLANS", {})
    layers = torch.nn.ModuleList(QuantLinear.from_weight(torch.randn(o, i, device="cuda") * 0.05)
                                 for o, i in ((256, 512), (128, 256), (256, 512)))
    plans = awq_module.tune(layers, rows=(1, 4, 32, 256), triton=triton, int4pack=int4pack)
    assert set(plans) == {(256, 512), (128, 256)} and awq_module.tuned_with_triton() == triton
    allowed = {"dense"} | ({"gemv", "gemm_t", "gemm"} if triton else set()) | ({"int4pack"} if int4pack else set())
    for layer in layers:
        plan = plans[(layer.out_features, layer.in_features)]
        kernels = {kernel for _, kernel, _ in plan}
        assert [upto for upto, _, _ in plan] == sorted({upto for upto, _, _ in plan}) and plan[-1][0] == 256
        assert kernels <= allowed
        assert (layer._int4pack is not None) == ("int4pack" in kernels)  # the second copy only where it is used
        reference = layer.dense_weight(torch.float32)
        for rows in (1, 3, 7, 9, 20, 300):
            x = torch.randn(rows, layer.in_features, device="cuda", dtype=torch.bfloat16)
            assert rel_err(layer(x), x.float() @ reference.t()) < 1e-2, rows


@needs_triton_gpu
@torch.no_grad()
def test_triton_rung_needs_a_triton_tuned_model(monkeypatch):
    monkeypatch.setattr(awq_module, "_PLANS", {})
    model = small_model().cuda()
    model.cosin_m_thetas = model.cosin_m_thetas.cuda()  # a plain attribute, so .cuda() skips it
    awq_quantize(model, randint(2, 32).cuda(), n_grid=3)
    awq_module.tune(model, rows=(1, 8), triton=False)  # rung awq's kernels only
    with pytest.raises(ValueError, match="triton=True"):
        Engine(model, "triton")
    awq_module.tune(model, rows=(1, 8))
    Engine(model, "triton")
    with pytest.raises(ValueError, match="nothing to tune"):
        awq_module.tune(model, triton=False, int4pack=False)


INTERPRETED = """
import torch
from vellum.mistral import w4a16
from vellum.mistral.awq import QuantLinear

torch.manual_seed(0)
layer = QuantLinear.from_weight(torch.randn(40, 256) * 0.05)
reference = layer.dense_weight(torch.float32)
sz = layer.sz.float()  # the kernels upcast the pairs anyway; fp32 spares the interpreter bf16
for kernel, cfg, rows in (
    ("gemv", dict(block_m=1, block_n=16, groups=2, num_warps=4, num_stages=1), 3),
    ("gemv", dict(block_m=4, block_n=16, groups=1, num_warps=4, num_stages=1), 6),
    ("gemm_t", dict(block_m=16, block_n=32, block_kh=32, num_warps=4, num_stages=1), 5),
    ("gemm", dict(block_m=16, block_n=32, block_kh=32, num_warps=4, num_stages=1), 5),
):
    x = torch.randn(rows, 256)
    got = w4a16.matmul(kernel, cfg, x, layer.nibbles, sz)
    want = x @ reference.t()
    err = ((got - want).norm() / want.norm()).item()
    assert got.shape == (rows, 40) and err < 5e-3, (kernel, err)  # loose enough for tf32 dots
"""


def test_triton_kernels_in_the_interpreter():
    """The kernels' indexing and dequantization, run by Triton's CPU interpreter in fp32: an
    indexing mistake shows up as an O(1) error, far past rounding."""
    pytest.importorskip("triton")
    done = subprocess.run([sys.executable, "-c", INTERPRETED], env={**os.environ, "TRITON_INTERPRET": "1"},
                          cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]


def test_rebuilt_weight_is_the_dequantized_weight():
    """The fused-friendly rebuild (nibbles -> fp32 math -> one rounding) == the reference."""
    from vellum.mistral.awq import _dense

    q, sz = quantize(torch.randn(256, 384) * 0.05)
    packed = pack_nibbles(q)
    assert torch.equal(_dense(packed, sz, torch.float32), dequantize(q, sz))
    # one rounding at the end: never more than half a bf16 step (8 significant bits) from exact
    exact = dequantize(q, sz).double()
    err = (_dense(packed, sz, torch.bfloat16).double() - exact).abs()
    assert (err <= exact.abs() * 2.0**-8 + 1e-12).all()


def test_compiled_rebuild_matches_eager():
    """torch.compile fuses the rebuild into one kernel (Triton on CUDA, C++ here) without changing it."""
    from vellum.mistral.awq import _dense

    q, sz = quantize(torch.randn(128, 256) * 0.05)
    packed = pack_nibbles(q)
    fused = torch.compile(_dense, dynamic=False)(packed, sz, torch.bfloat16)
    eager = _dense(packed, sz, torch.bfloat16)
    step = eager.float().abs() * 2.0**-7 + 1e-12  # one bf16 step
    assert ((fused.float() - eager.float()).abs() <= step).all()


@torch.no_grad()
def test_folding_scales_is_exact():
    model = small_model()
    layer = model.layers[0]
    x = torch.randn(2, 7, SMALL["dim"])
    a = layer.attention
    before = [lin(layer.attention_norm(x)) for lin in (a.wq, a.wk, a.wv)]
    s = torch.rand(SMALL["dim"]) * 4 + 0.25
    _fold(s, [a.wq, a.wk, a.wv], layer.attention_norm)
    for lin, want in zip((a.wq, a.wk, a.wv), before):
        torch.testing.assert_close(lin(layer.attention_norm(x)), want, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_scale_search_beats_round_to_nearest_on_outlier_channels():
    x = outlier_inputs()
    w = torch.randn(128, 256) * 0.05
    s = search_scale(x, [w])
    awq, rtn = _output_error(x, w, s), _output_error(x, w, torch.ones(256))
    assert awq < 0.8 * rtn, (awq, rtn)
    calm = torch.randn(512, 256)  # no outliers: never worse than RTN (alpha = 0 is on the grid)
    assert _output_error(calm, w, search_scale(calm, [w])) <= _output_error(calm, w, torch.ones(256))


@torch.no_grad()
def test_awq_model_tracks_full_precision_and_beats_rtn():
    fp = small_model()
    calibration, held_out = randint(4, 32), randint(2, 24, seed=7)
    fp.allocate_cache(*held_out.shape)
    want = fp(held_out, torch.arange(held_out.shape[1]))
    errors = {}
    for name, search in (("awq", True), ("rtn", False)):
        model = small_model()
        awq_quantize(model, calibration, search=search, n_grid=10)
        assert sum(isinstance(m, QuantLinear) for m in model.modules()) == 7 * SMALL["n_layers"]
        model.allocate_cache(*held_out.shape)
        got = model(held_out, torch.arange(held_out.shape[1]))
        errors[name] = ((got - want).norm() / want.norm()).item()
    assert errors["awq"] < 0.25, errors  # 4-bit, but still the same model
    assert errors["awq"] <= errors["rtn"] * 1.05, errors


@torch.no_grad()
def test_quantized_checkpoint_round_trips(tmp_path):
    model = small_model()
    awq_quantize(model, randint(2, 32), n_grid=5)
    save_quantized(model, tmp_path, {"method": "awq"})
    assert json.loads((tmp_path / "quantization.json").read_text())["method"] == "awq"
    loaded = load_model(tmp_path, device="cpu")  # bf16, the dtype the 4-bit paths compute in
    assert next(loaded.parameters()).dtype == torch.bfloat16
    for a, b in zip(model.modules(), loaded.modules()):
        if isinstance(a, QuantLinear):
            assert b.nibbles.dtype == torch.uint8 and b.sz.dtype == torch.bfloat16  # 4 bits stay 4 bits
            assert torch.equal(a.nibbles, b.nibbles) and torch.equal(a.sz, b.sz)
            assert set(dict(b.named_buffers())) == {"nibbles", "sz"}  # and are the only copy
    prompt, fed = randint(2, 6), randint(2, 3, seed=2)
    engine = Engine(loaded, "chunked", prefill_tokens=4)  # the awq rung's path, short of CUDA graphs
    engine.setup(2, 6 + 3)
    logits = [engine.prefill(prompt)] + [engine.decode(fed[:, i], 6 + i) for i in range(3)]
    assert all(torch.isfinite(t).all() for t in logits)


@torch.no_grad()
def test_perplexity_is_exp_mean_next_token_loss():
    model = small_model()
    ids = randint(40)
    logits = model(ids[None, :-1], torch.arange(39))[0]
    want = torch.nn.functional.cross_entropy(logits, ids[1:]).exp().item()
    assert perplexity(model, ids, window=64) == pytest.approx(want, rel=1e-5)
    windows = perplexity(model, ids, window=16)  # 3 windows, each read from scratch
    assert windows == pytest.approx(windows) and windows > 1


def test_quantize_cli_end_to_end(tmp_path, monkeypatch):
    """`vellum quantize` with its downloads faked: tokenizer and WikiText-2 stand-ins."""
    from safetensors.torch import save_file

    import vellum.mistral.awq as awq_module
    import vellum.mistral as mistral
    from vellum.cli import main

    src = tmp_path / "src"
    src.mkdir()
    model = small_model()
    save_file(model.state_dict(), str(src / "consolidated.safetensors"))
    (src / "params.json").write_text(json.dumps({**SMALL, "norm_eps": 1e-5, "rope_theta": 1e6}))

    class Tokenizer:
        def encode(self, text, bos, eos):
            return [ord(c) % SMALL["vocab_size"] for c in text]

    monkeypatch.setattr(mistral, "load_tokenizer", lambda source: Tokenizer())
    monkeypatch.setattr(awq_module, "wikitext", lambda split: "the quick brown fox jumps over the lazy dog. " * 40)
    out, quality = tmp_path / "awq", tmp_path / "quality.json"
    main(["quantize", "--model", str(src), "--out", str(out), "--device", "cpu", "--samples", "4",
          "--length", "32", "--grid", "5", "--quality", str(quality)])
    report = json.loads(quality.read_text())
    assert set(report["perplexity"]) == {"bf16", "awq", "rtn"}
    assert all(v > 1 for v in report["perplexity"].values())
    assert isinstance(load_model(out, device="cpu").layers[0].attention.wq, QuantLinear)
