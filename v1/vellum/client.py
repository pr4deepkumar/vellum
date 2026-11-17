"""Async load generator for OpenAI-compatible completion servers (vLLM).

Each request is streamed so we can timestamp individual tokens:

    TTFT  time to first token        (prefill + queueing)
    ITL   inter-token latency        (gap between consecutive streamed chunks)
    TPOT  time per output token      (e2e - ttft) / (output_tokens - 1)
    E2E   end-to-end request latency

Two load modes:
    closed loop  -- fixed number of in-flight requests ("concurrency"), which is
                    the client-side analogue of batch size for a continuously
                    batching server.
    open loop    -- Poisson arrivals at a target request rate, which exposes
                    queueing behaviour once the server saturates.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field

import aiohttp
import numpy as np

# Common words; for most BPE tokenizers each is ~1 token with a leading space.
_WORDS = (
    "time person year way day thing man world life hand part child eye woman place "
    "work week case point government company number group problem fact system "
    "program question night story water room mother area money month lot right study "
    "book job word business issue side kind head house service friend father power "
    "hour game line end member law car city community name president team minute idea"
).split()


def make_prompt(n_words: int, rng: random.Random) -> str:
    """Random word salad of roughly `n_words` tokens.

    Every prompt is unique so vLLM's automatic prefix cache cannot short-circuit
    prefill and inflate throughput numbers.
    """
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


@dataclass
class RequestResult:
    ok: bool
    start: float
    end: float
    ttft: float | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    itls: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def e2e(self) -> float:
        return self.end - self.start

    @property
    def tpot(self) -> float | None:
        if self.ttft is None or self.output_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # vLLM extension: force exactly max_tokens so output length is controlled.
        "ignore_eos": True,
    }
    start = time.perf_counter()
    res = RequestResult(ok=False, start=start, end=start)
    last = start
    n_chunks = 0
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                res.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                res.end = time.perf_counter()
                return res
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                now = time.perf_counter()
                if chunk.get("usage"):
                    res.prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    res.output_tokens = chunk["usage"].get("completion_tokens", 0)
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    if res.ttft is None:
                        res.ttft = now - start
                    else:
                        res.itls.append(now - last)
                    last = now
                    n_chunks += 1
        res.end = time.perf_counter()
        if not res.output_tokens:  # server didn't send usage; fall back to chunk count
            res.output_tokens = n_chunks
        res.ok = res.ttft is not None
        if not res.ok:
            res.error = "no tokens received"
    except Exception as e:  # noqa: BLE001 -- record any transport failure as a failed request
        res.end = time.perf_counter()
        res.error = f"{type(e).__name__}: {e}"
    return res


# --------------------------------------------------------------------------- #
# Server-side metrics (Prometheus /metrics) sampled during a run
# --------------------------------------------------------------------------- #

_METRIC_RE = re.compile(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$")
_GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage",  # vLLM V1 name
    "vllm:gpu_cache_usage_perc": "kv_cache_usage",  # legacy name
}


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC_RE.match(line.strip())
        if m and m.group(1) in _GAUGES:
            key = _GAUGES[m.group(1)]
            out[key] = out.get(key, 0.0) + float(m.group(3))  # sum across engines
    return out


async def poll_metrics(
    session: aiohttp.ClientSession, base_url: str, stop: asyncio.Event, samples: list[dict]
) -> None:
    while not stop.is_set():
        try:
            async with session.get(f"{base_url}/metrics") as resp:
                if resp.status == 200:
                    samples.append(parse_metrics(await resp.text()))
        except Exception:  # noqa: BLE001 -- metrics are best-effort
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass


# --------------------------------------------------------------------------- #
# Running one load level
# --------------------------------------------------------------------------- #


async def run_level(
    base_url: str,
    model: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    concurrency: int | None = None,
    request_rate: float | None = None,
    shared_prefix_len: int = 0,
    seed: int = 0,
    timeout_s: float = 600.0,
) -> tuple[list[RequestResult], list[dict], float]:
    """Run one benchmark level. Returns (per-request results, metric samples, wall time)."""
    if (concurrency is None) == (request_rate is None):
        raise ValueError("specify exactly one of concurrency or request_rate")
    rng = random.Random(seed)
    # A shared prefix models a common system prompt / few-shot header: identical
    # leading tokens across every request (and every level), which vLLM's prefix
    # cache can reuse. Only the suffix varies with `seed`.
    prefix = make_prompt(shared_prefix_len, random.Random(-1)) + " " if shared_prefix_len else ""
    unique = max(1, input_len - shared_prefix_len)
    prompts = [prefix + make_prompt(unique, rng) for _ in range(num_requests)]

    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        stop = asyncio.Event()
        samples: list[dict] = []
        poller = asyncio.create_task(poll_metrics(session, base_url, stop, samples))

        t0 = time.perf_counter()
        if concurrency is not None:
            sem = asyncio.Semaphore(concurrency)

            async def bounded(p: str) -> RequestResult:
                async with sem:
                    return await send_request(session, base_url, model, p, output_len)

            results = await asyncio.gather(*(bounded(p) for p in prompts))
        else:
            tasks = []
            for p in prompts:
                tasks.append(asyncio.create_task(send_request(session, base_url, model, p, output_len)))
                await asyncio.sleep(rng.expovariate(request_rate))
            results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

        stop.set()
        await poller
    return list(results), samples, wall


def _pct(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def summarize(
    results: list[RequestResult], samples: list[dict], wall: float, **meta
) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft for r in ok]
    tpot = [r.tpot for r in ok if r.tpot is not None]
    itl = [x for r in ok for x in r.itls]
    e2e = [r.e2e for r in ok]
    out_tok = sum(r.output_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)

    s = {
        **meta,
        "num_requests": len(results),
        "num_ok": len(ok),
        "num_failed": len(results) - len(ok),
        "duration_s": wall,
        "request_throughput": len(ok) / wall,
        "output_token_throughput": out_tok / wall,
        "total_token_throughput": (in_tok + out_tok) / wall,
        "mean_prompt_tokens": in_tok / len(ok) if ok else 0,
        "mean_output_tokens": out_tok / len(ok) if ok else 0,
    }
    for name, xs in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2e", e2e)):
        s[f"mean_{name}_ms"] = float(np.mean(xs)) * 1e3 if xs else None
        for q in (50, 90, 99):
            v = _pct(xs, q)
            s[f"p{q}_{name}_ms"] = v * 1e3 if v is not None else None
    for key in ("running", "waiting", "kv_cache_usage"):
        vals = [x[key] for x in samples if key in x]
        s[f"max_{key}"] = max(vals) if vals else None
        s[f"mean_{key}"] = float(np.mean(vals)) if vals else None
    errors = [r.error for r in results if r.error]
    if errors:
        s["sample_errors"] = errors[:3]
    return s


def results_to_rows(results: list[RequestResult]) -> list[dict]:
    rows = []
    for r in results:
        d = asdict(r)
        d.pop("itls")
        d["e2e"] = r.e2e
        d["tpot"] = r.tpot
        rows.append(d)
    return rows
