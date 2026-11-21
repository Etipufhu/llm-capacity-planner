import asyncio
import json

import httpx

from capplan.client import build_payload, parse_sse_line, run_load
from capplan.stats import summarize


def sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def token_chunk(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def fake_server(n_tokens=5, delay=0.01, fail_every=0):
    """Mock OpenAI streaming endpoint that sends one token every `delay` seconds."""
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if fail_every and len(seen) % fail_every == 0:
            return httpx.Response(500, text="boom")

        async def stream():
            yield b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
            for i in range(n_tokens):
                await asyncio.sleep(delay)
                yield sse([token_chunk(f"t{i} ")]).split("data: [DONE]")[0].encode()
            usage = {"prompt_tokens": 42, "completion_tokens": n_tokens}
            yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, content=stream())

    return httpx.MockTransport(handler), seen


PROMPTS = [{"id": "p0", "messages": [{"role": "user", "content": "hi"}]}]


def test_parse_sse_line():
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line(": keep-alive") is None
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}


def test_build_payload_uses_prompt_max_tokens_and_extra_body():
    p = {"messages": [], "max_tokens": 50}
    body = build_payload("m", p, 300, 0.5, {"reasoning_effort": "none"})
    assert body["max_tokens"] == 50
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["reasoning_effort"] == "none"


def test_run_load_measures_tokens_and_latency():
    transport, seen = fake_server(n_tokens=5, delay=0.02)
    results, wall = asyncio.run(run_load("http://x/v1", "m", PROMPTS, 2, 4, transport=transport))
    assert len(results) == 4 and len(seen) == 4
    for r in results:
        assert r.ok
        assert r.output_tokens == 5
        assert r.prompt_tokens == 42
        assert r.ttft >= 0.015
        assert 0.01 < r.itl_mean < 0.1
        assert r.e2e >= r.ttft
    s = summarize(results, wall, 2)
    assert s["errors"] == 0
    assert s["output_tok_per_s"] > 0
    assert s["user_tok_per_s_p50"] > 10


def test_run_load_keeps_concurrency_in_flight():
    transport, _ = fake_server(n_tokens=3, delay=0.05)
    results, wall = asyncio.run(run_load("http://x/v1", "m", PROMPTS, 4, 8, transport=transport))
    # 8 requests of ~0.15 s each with 4 in flight take ~2 rounds, not 8
    assert wall < 0.15 * 8 * 0.6
    assert {r.worker for r in results} == {0, 1, 2, 3}


def test_http_errors_are_recorded_not_raised():
    transport, _ = fake_server(fail_every=2)
    results, wall = asyncio.run(run_load("http://x/v1", "m", PROMPTS, 1, 4, transport=transport))
    errors = [r for r in results if not r.ok]
    assert len(errors) == 2
    assert errors[0].error.startswith("HTTP 500")
    assert summarize(results, wall, 1)["error_rate"] == 0.5
