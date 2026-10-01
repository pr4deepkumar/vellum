import asyncio
import json
import random

import pytest
from aiohttp.test_utils import TestServer

from mock_server import make_app
from vellum.cli import main
from vellum.client import RequestResult, make_prompt, parse_metrics, run_level, summarize
from vellum.report import cost_per_million, report_dir


def test_tpot_excludes_first_token():
    r = RequestResult(ok=True, start=0.0, end=1.1, ttft=0.1, output_tokens=11)
    assert r.e2e == pytest.approx(1.1)
    assert r.tpot == pytest.approx(0.1)  # 1.0 s spread over the 10 tokens after the first


def test_prompts_are_unique_and_sized():
    rng = random.Random(0)
    a, b = make_prompt(100, rng), make_prompt(100, rng)
    assert a != b and len(a.split()) == 100


def test_parse_metrics_handles_labels_and_both_kv_names():
    text = (
        "# HELP vllm:num_requests_running x\n"
        'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
        'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25\n'
        'vllm:gpu_cache_usage_perc{model_name="m"} 0.5\n'
        "vllm:some_histogram_bucket{le=\"1\"} 9\n"
    )
    m = parse_metrics(text)
    assert m["running"] == 3.0
    assert m["kv_cache_usage"] == pytest.approx(0.75)
    assert "waiting" not in m


def test_cost_per_million():
    # 1000 tok/s on a $3.60/hr GPU -> 3.6M tok/hr -> $1.00 per 1M
    assert cost_per_million(1000, 3.6) == pytest.approx(1.0)
    assert cost_per_million(1000, None) is None


async def test_run_level_against_mock():
    async with TestServer(make_app()) as srv:
        base = str(srv.make_url("")).rstrip("/")
        results, samples, wall = await run_level(base, "mock/model", num_requests=12,
                                                 input_len=64, output_len=16, concurrency=4)
        s = summarize(results, samples, wall, label="t", mode="concurrency", level=4)
    assert s["num_ok"] == 12 and s["num_failed"] == 0
    assert s["mean_output_tokens"] == 16
    assert s["mean_prompt_tokens"] == 64
    assert s["p50_ttft_ms"] > 0 and s["p50_tpot_ms"] > 0
    assert s["max_running"] is not None


async def test_shared_prefix_is_shared():
    async with TestServer(make_app()) as srv:
        base = str(srv.make_url("")).rstrip("/")
        results, _, _ = await run_level(base, "mock/model", num_requests=4, input_len=100,
                                        output_len=4, concurrency=1, shared_prefix_len=60)
    assert all(r.prompt_tokens == 100 for r in results)


def test_cli_end_to_end(tmp_path, monkeypatch):
    import threading, asyncio
    from aiohttp import web

    loop = asyncio.new_event_loop()
    runner = web.AppRunner(make_app())
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = site._server.sockets[0].getsockname()[1]
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        for label in ("fp16", "awq"):
            main(["sweep", "--base-url", f"http://127.0.0.1:{port}", "--label", label,
                  "--concurrency", "1", "4", "--input-len", "32", "--output-len", "8",
                  "--min-requests", "8", "--out", str(tmp_path)])
    finally:
        loop.call_soon_threadsafe(loop.stop)
    data = json.loads((tmp_path / "fp16.json").read_text())
    assert [lv["level"] for lv in data["levels"]] == [1, 4]
    md = report_dir(tmp_path, gpu_cost_per_hour=0.8)
    assert "Head-to-head at c=4" in md and "$/1M out" in md
    main(["plot", str(tmp_path)])
    assert (tmp_path / "main_concurrency.png").exists()


async def test_levels_get_fresh_prompts_but_same_prefix(monkeypatch):
    import vellum.client as c
    seen = []

    async def fake_send(session, base_url, model, prompt, max_tokens):
        seen.append(prompt)
        return RequestResult(ok=True, start=0, end=0, ttft=0)

    monkeypatch.setattr(c, "send_request", fake_send)
    monkeypatch.setattr(c, "poll_metrics", lambda *a: asyncio.sleep(0))
    for seed in (1, 2):
        await c.run_level("http://x", "m", 3, 50, 4, concurrency=1, shared_prefix_len=30, seed=seed)
    assert len(set(seen)) == 6  # no prompt repeats across levels
    assert len({" ".join(p.split()[:30]) for p in seen}) == 1  # one shared prefix


def test_offline_results_render(tmp_path):
    from vellum.offline import _summary

    for label, step in (("hf-fp16", 0.030), ("vllm-fp16", 0.012)):
        levels = [_summary(label, "m", bs, 512, 128, 0.05 * bs, 0.05 * bs + 127 * step * (1 + bs / 32))
                  for bs in (1, 4, 16)]
        levels.append(_summary(label, "m", 64, 512, 128, 0, 0, failed=True))
        meta = {"label": label, "experiment": "engine", "model": "m"}
        (tmp_path / f"{label}.json").write_text(json.dumps({"meta": meta, "levels": levels}))
    s = levels[0]
    assert s["output_token_throughput"] == pytest.approx(128 / (0.05 + 127 * 0.012 * (1 + 1 / 32)))
    md = report_dir(tmp_path, 1.0)
    assert "[engine] Head-to-head at bs=16" in md  # bs=64 failed, so it is excluded
    main(["plot", str(tmp_path)])
    assert (tmp_path / "engine_batch_size.png").exists()
