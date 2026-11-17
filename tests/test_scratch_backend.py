"""`vellum offline --backend scratch` end to end on CPU: a tiny random Mistral saved the way
Mistral ships 7B, through the CLI, into JSON, the report and the engine chart. Needs torch."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from vellum.cli import main  # noqa: E402
from vellum.offline import SCRATCH_RUNGS  # noqa: E402
from vellum.report import report_dir  # noqa: E402


def _tiny_checkpoint(folder):
    from safetensors.torch import save_file
    from vellum.mistral import Transformer, TransformerArgs

    params = dict(dim=64, n_layers=2, head_dim=16, hidden_dim=128, n_heads=4, n_kv_heads=2,
                  norm_eps=1e-5, vocab_size=128, rope_theta=1e6)  # params.json's own keys
    torch.manual_seed(0)
    model = Transformer(TransformerArgs(**{k: v for k, v in params.items() if k != "rope_theta"}))
    folder.mkdir()
    save_file(model.state_dict(), str(folder / "consolidated.safetensors"))
    (folder / "params.json").write_text(json.dumps(params))
    return folder


def test_cli_rungs_match_the_engine_ladder():
    from vellum.mistral import LADDER

    assert SCRATCH_RUNGS == LADDER


def test_single_copy_is_rung_triton_only(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path / "ckpt")
    with pytest.raises(SystemExit, match="rung triton"):
        main(["offline", "--backend", "scratch", "--model", str(ckpt), "--device", "cpu", "--rung", "awq",
              "--awq-single-copy", "--label", "x", "--batch-sizes", "1", "--out", str(tmp_path / "out")])


def test_scratch_backend_end_to_end(tmp_path):
    ckpt, out = _tiny_checkpoint(tmp_path / "ckpt"), tmp_path / "results"
    for rung in ("naive", "gqa", "sdpa", "chunked"):
        main(["offline", "--backend", "scratch", "--model", str(ckpt), "--device", "cpu", "--rung", rung,
              "--label", f"scratch-{rung}", "--batch-sizes", "1", "2", "4", "--input-len", "16",
              "--output-len", "8", "--repeats", "1", "--out", str(out)])
    data = json.loads((out / "scratch-sdpa.json").read_text())
    assert data["meta"]["backend"] == "scratch"
    assert data["meta"]["server_args"] == "offline scratch rung=sdpa"
    assert [lv["level"] for lv in data["levels"]] == [1, 2, 4]
    assert all(lv["num_ok"] == lv["level"] and lv["output_token_throughput"] > 0 for lv in data["levels"])
    md = report_dir(out, 1.0)
    assert "[engine] Head-to-head at bs=4 (relative to scratch-naive)" in md  # ladder order
    main(["plot", str(out)])
    assert (out / "engine_batch_size.png").exists()  # the top rung beside any reference engines
    assert (out / "engine_ladder.png").exists()  # rung by rung
