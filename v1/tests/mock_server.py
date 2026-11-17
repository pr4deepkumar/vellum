"""Tiny fake vLLM server for testing the harness without a GPU.

Speaks enough of the OpenAI completions streaming protocol (+ /metrics) for the
client. Latency is a toy model, NOT representative of real hardware.

    python tests/mock_server.py --port 8011
"""

from __future__ import annotations

import argparse
import asyncio
import json

from aiohttp import web

STATE = {"running": 0, "prefixes": set()}
MODEL = "mock/model"


async def models(_):
    return web.json_response({"object": "list", "data": [{"id": MODEL, "object": "model"}]})


async def health(_):
    return web.Response(text="ok")


async def metrics(_):
    r = STATE["running"]
    body = (
        f'vllm:num_requests_running{{model_name="{MODEL}"}} {float(r)}\n'
        f'vllm:num_requests_waiting{{model_name="{MODEL}"}} 0.0\n'
        f'vllm:kv_cache_usage_perc{{model_name="{MODEL}"}} {min(1.0, r / 64):.4f}\n'
    )
    return web.Response(text=body)


async def completions(request: web.Request):
    body = await request.json()
    words = body["prompt"].split()
    n_out = body["max_tokens"]
    key = " ".join(words[:32])
    cached = key in STATE["prefixes"]
    STATE["prefixes"].add(key)

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    STATE["running"] += 1
    try:
        prefill = 0.002 + (0.0 if cached else 0.00002 * len(words))
        await asyncio.sleep(prefill * (1 + 0.05 * STATE["running"]))
        for i in range(n_out):
            chunk = {"id": "x", "object": "text_completion", "model": MODEL,
                     "choices": [{"index": 0, "text": f" t{i}", "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(0.001 * (1 + 0.03 * STATE["running"]))
        usage = {"id": "x", "object": "text_completion", "model": MODEL, "choices": [],
                 "usage": {"prompt_tokens": len(words), "completion_tokens": n_out,
                           "total_tokens": len(words) + n_out}}
        await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
    finally:
        STATE["running"] -= 1
    return resp


def make_app() -> web.Application:
    app = web.Application()
    app.add_routes([web.get("/v1/models", models), web.get("/health", health),
                    web.get("/metrics", metrics), web.post("/v1/completions", completions)])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    web.run_app(make_app(), port=ap.parse_args().port)
