import json

import pytest

from capplan.gpu import parse_smi_line
from capplan.stats import percentile, summarize
from capplan.sweep import deep_merge, load_config, num_requests, write_summary
from capplan.workloads import brief_prompts, chat_prompts


def test_percentile_matches_linear_interpolation():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(xs, 50) == pytest.approx(5.5)
    assert percentile(xs, 95) == pytest.approx(9.55)
    assert percentile([3.0], 99) == 3.0
    assert percentile([], 50) is None


def test_summarize_ignores_failed_requests_in_latency():
    ok = {"error": None, "ttft": 0.2, "e2e": 2.0, "output_tokens": 100, "prompt_tokens": 10}
    ok["itl_mean"] = 0.02
    bad = {"error": "HTTP 500", "ttft": None, "e2e": None, "output_tokens": 0}
    bad["prompt_tokens"] = None
    bad["itl_mean"] = None
    s = summarize([ok, ok, bad], wall=4.0, concurrency=2)
    assert s["errors"] == 1
    assert s["error_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert s["output_tok_per_s"] == 50.0
    assert s["ttft_p95_ms"] == 200.0
    assert s["user_tok_per_s_p50"] == 50.0


def test_parse_smi_line():
    s = parse_smi_line("1736, 8151, 12, 9.81")
    assert s == {"mem_used_mib": 1736, "mem_total_mib": 8151, "util_pct": 12, "power_w": 9.81}
    assert parse_smi_line("1736, 8151, 12, [N/A]")["power_w"] is None
    assert parse_smi_line("garbage") is None


def test_deep_merge_and_config(tmp_path):
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}
    cfg = tmp_path / "x.yaml"
    cfg.write_text(
        "defaults:\n  server:\n    env: {OLLAMA_NUM_PARALLEL: 8}\n"
        "runs:\n  - {name: a, model: m1}\n"
        "  - {name: b, model: m2, server: {env: {OLLAMA_KV_CACHE_TYPE: q8_0}}}\n"
    )
    runs = load_config(cfg)
    assert runs[0]["server"]["env"] == {"OLLAMA_NUM_PARALLEL": 8}
    assert runs[1]["server"]["env"] == {"OLLAMA_NUM_PARALLEL": 8, "OLLAMA_KV_CACHE_TYPE": "q8_0"}
    assert runs[1]["concurrency"] == [1, 2, 4, 8, 16, 32]


def test_num_requests():
    assert num_requests(1, 4, 16) == 16
    assert num_requests(32, 4, 16) == 128


def test_workloads():
    chat = chat_prompts(10)
    assert len(chat) == 10 and chat[0]["messages"][0]["role"] == "system"
    assert chat == chat_prompts(10)  # deterministic
    briefs = brief_prompts()
    assert len(briefs) == 154
    assert all(m["role"] != "assistant" for b in briefs for m in b["messages"])


def test_write_summary(tmp_path):
    raw = tmp_path / "raw" / "q4" / "c004.json"
    raw.parent.mkdir(parents=True)
    rec = {
        "meta": {"run": "q4", "model": "m", "server": {"env": {"OLLAMA_NUM_PARALLEL": 4}}},
        "summary": {"concurrency": 4, "output_tok_per_s": 100.0},
        "gpu": {"vram_peak_mib": 3000},
        "requests": [],
    }
    raw.write_text(json.dumps(rec))
    path = write_summary(tmp_path)
    text = path.read_text()
    assert "num_parallel" in text.splitlines()[0]
    assert "q4" in text and "3000" in text


def test_match_parallel_expands_one_server_per_level():
    from capplan.sweep import DEFAULTS, expand_matched

    cfg = deep_merge(DEFAULTS, {"name": "t", "model": "m", "match_parallel": True})
    cfg["concurrency"] = [1, 4]
    subs = expand_matched(cfg)
    assert [s["concurrency"] for s in subs] == [[1], [4]]
    assert [s["server"]["env"]["OLLAMA_NUM_PARALLEL"] for s in subs] == [1, 4]
    assert subs[1]["log_name"] == "t-np4" and subs[1]["name"] == "t"
    assert expand_matched({**cfg, "match_parallel": False}) == [{**cfg, "match_parallel": False}]
