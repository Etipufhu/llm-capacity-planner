"""Closed-loop streaming load generator for OpenAI-compatible chat endpoints.

`concurrency` workers share one queue of prompts. Each worker sends a request,
reads the SSE stream to the end and immediately takes the next prompt, so the
server always sees `concurrency` requests in flight (minus the drain at the end).
Timestamps come from `time.perf_counter()` and are stored relative to the start
of the run.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import httpx


@dataclass
class RequestResult:
    id: int
    worker: int
    prompt_id: str
    t_start: float
    ttft: float | None = None
    e2e: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int = 0
    itl_mean: float | None = None
    itl_max: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.ttft is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_text(chunk: dict[str, Any]) -> str:
    """Text carried by one streamed chunk (answer or reasoning)."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    parts = [delta.get(k) for k in ("content", "reasoning_content", "reasoning")]
    return "".join(p for p in parts if isinstance(p, str))


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Return the JSON payload of a `data:` line, None for anything else or [DONE]."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    return json.loads(data)


async def stream_one(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    result: RequestResult,
    t0: float,
) -> RequestResult:
    """Send one streaming request and fill in the timing fields of `result`."""
    start = time.perf_counter()
    result.t_start = start - t0
    stamps: list[float] = []
    usage: dict[str, Any] | None = None
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:200]
                result.error = f"HTTP {resp.status_code}: {body}"
                return result
            async for line in resp.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                if chunk.get("error"):
                    result.error = str(chunk["error"])[:200]
                    return result
                if _token_text(chunk):
                    stamps.append(time.perf_counter())
                if chunk.get("usage"):
                    usage = chunk["usage"]
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:200]
        return result

    end = time.perf_counter()
    result.e2e = end - start
    if not stamps:
        result.error = "no tokens received"
        return result
    result.ttft = stamps[0] - start
    n_out = len(stamps)
    if usage:
        n_out = int(usage.get("completion_tokens") or n_out)
        if usage.get("prompt_tokens") is not None:
            result.prompt_tokens = int(usage["prompt_tokens"])
    result.output_tokens = n_out
    if n_out > 1 and stamps[-1] > stamps[0]:
        # Mean time per token after the first one. Using the token count (not
        # the chunk count) keeps this right when a server packs several tokens
        # into one chunk.
        result.itl_mean = (stamps[-1] - stamps[0]) / (n_out - 1)
        result.itl_max = max(b - a for a, b in zip(stamps, stamps[1:], strict=False))
    return result


def build_payload(
    model: str,
    prompt: dict[str, Any],
    max_tokens: int,
    temperature: float,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": prompt["messages"],
        "max_tokens": int(prompt.get("max_tokens") or max_tokens),
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        payload.update(extra_body)
    return payload


async def run_load(
    base_url: str,
    model: str,
    prompts: list[dict[str, Any]],
    concurrency: int,
    num_requests: int,
    max_tokens: int = 256,
    temperature: float = 0.7,
    extra_body: dict[str, Any] | None = None,
    timeout: float = 300.0,
    on_done: Callable[[RequestResult], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[RequestResult], float]:
    """Run `num_requests` requests with `concurrency` workers.

    Prompts are used in order and wrap around. Returns the results (in request
    order) and the wall time from the first send to the last response.
    """
    if concurrency < 1 or num_requests < 1:
        raise ValueError("concurrency and num_requests must be >= 1")
    url = base_url.rstrip("/") + "/chat/completions"
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(num_requests):
        queue.put_nowait(i)
    results: list[RequestResult | None] = [None] * num_requests

    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    t_out = httpx.Timeout(timeout, connect=30.0)
    async with httpx.AsyncClient(limits=limits, timeout=t_out, transport=transport) as client:
        t0 = time.perf_counter()

        async def worker(w: int) -> None:
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                prompt = prompts[i % len(prompts)]
                payload = build_payload(model, prompt, max_tokens, temperature, extra_body)
                res = RequestResult(id=i, worker=w, prompt_id=str(prompt.get("id", i)), t_start=0)
                await stream_one(client, url, payload, res, t0)
                results[i] = res
                if on_done:
                    on_done(res)

        await asyncio.gather(*(worker(w) for w in range(concurrency)))
        wall = time.perf_counter() - t0
    return [r for r in results if r is not None], wall
