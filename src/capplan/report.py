"""Turn results/summary.csv into the capacity table, the charts and the calculator data."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .capacity import (
    GPU,
    SLO,
    estimate_gpu,
    fit_curves,
    kv_bytes_per_token,
    max_concurrency,
    plan,
    vram_concurrency,
)

# Qwen3-1.7B: 28 layers, 8 KV heads of 128. Q8_0 stores 32 values in 34 bytes.
KV_LAYOUT = {"layers": 28, "kv_heads": 8, "head_dim": 128}
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}

NUM_FIELDS = {
    "concurrency",
    "requests",
    "errors",
    "num_parallel",
    "context_length",
    "model_vram_mib",
    "gpu_samples",
    "max_tokens",
}

WORKLOADS = {
    "brief": {
        "run": "brief-tuned",
        "label": "Weekly brief (PR Radar model, ~740 tokens in, ~145 out)",
        "weights_gb": 1.9,  # GGUF file size, rounded up
    },
    "chat": {
        "run": "chat-tuned",
        "label": "Chat assistant (~60 tokens in, ~175 out)",
        "weights_gb": 2.2,
    },
}


def _num(v: str) -> Any:
    if v in ("", "None"):
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() else f


def load_summary(path: str | Path = "results/summary.csv") -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def rows_for(summary: list[dict[str, Any]], run: str) -> list[dict[str, Any]]:
    return sorted((r for r in summary if r["run"] == run), key=lambda r: r["concurrency"])


def load_gpus(path: str | Path = "configs/gpus.yaml") -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gpus = []
    for g in cfg["cloud"]:
        gpus.append(
            {
                "name": g["name"],
                "bandwidth_gbs": g["bandwidth_gbs"],
                "vram_gb": g["vram_gb"],
                "price_per_hour": round(g["instance_price_per_hour"] / g["gpus_per_instance"], 4),
                "instance": g["instance"],
                "gpus_per_instance": g["gpus_per_instance"],
                "source": g["source"],
            }
        )
    return {
        "measured": cfg["measured"],
        "gpus": gpus,
        "price_date": cfg["price_date"],
        "price_note": cfg["price_note"],
    }


def workload_model(rows: list[dict[str, Any]], weights_gb: float) -> dict[str, Any]:
    """Everything the calculator needs about one measured workload."""
    first = rows[0]
    kv_type = first.get("kv_cache_type") or "f16"
    slots = {r.get("num_parallel") for r in rows}
    matched = all(r.get("num_parallel") == r["concurrency"] for r in rows)
    if matched:
        slots_note = "as many parallel slots as concurrent requests"
    elif len(slots) == 1:
        slots_note = f"{first.get('num_parallel')} parallel slots"
    else:
        slots_note = "mixed slot counts"
    return {
        "model": first["model"],
        "slots_note": slots_note,
        "context_length": first.get("context_length"),
        "kv_cache_type": kv_type,
        "kv_bytes_token": kv_bytes_per_token(**KV_LAYOUT, bytes_per_elem=KV_BYTES[kv_type]),
        "weights_gb": weights_gb,
        "fits": fit_curves(rows),
        "rows": [
            {
                k: r.get(k)
                for k in (
                    "concurrency",
                    "ttft_p50_ms",
                    "ttft_p95_ms",
                    "itl_p50_ms",
                    "itl_p95_ms",
                    "e2e_p50_s",
                    "error_rate",
                    "output_tok_per_s",
                    "user_tok_per_s_p5",
                    "mean_prompt_tokens",
                    "mean_output_tokens",
                    "vram_peak_mib",
                )
            }
            for r in rows
        ],
    }


def capacity_table(
    wl: dict[str, Any], gpus: dict[str, Any], slo: SLO, users: int, think_s: float, headroom: float
) -> list[dict[str, Any]]:
    """One row per GPU: max concurrency, users per GPU, GPUs and monthly cost."""
    rows = wl["rows"]
    out = []
    measured = gpus["measured"]
    c_meas = max_concurrency(rows, slo)
    e2e_meas = _interp_e2e(rows, max(c_meas, 1))
    p = plan(users, c_meas, e2e_meas, think_s, None, headroom)
    top = max(r["concurrency"] for r in rows)
    out.append(
        {
            "gpu": measured["name"],
            "kind": "measured",
            "max_concurrency": int(c_meas),
            # all measured levels pass: the real limit is higher than we tested
            "limited_by": "not reached" if c_meas >= top else "measured",
            "e2e_p50_s": e2e_meas,
            **p,
        }
    )
    ref_bw = measured["bandwidth_gbs"]
    for g in gpus["gpus"]:
        gpu = GPU(g["name"], g["bandwidth_gbs"], g["vram_gb"], g["price_per_hour"])
        cap = vram_concurrency(gpu, wl["weights_gb"], wl["kv_bytes_token"], wl["context_length"])
        est = estimate_gpu(gpu, wl["fits"], ref_bw, slo, vram_cap=cap)
        p = plan(
            users, est["max_concurrency"], est["e2e_p50_s"], think_s, g["price_per_hour"], headroom
        )
        out.append(
            {
                "gpu": g["name"],
                "kind": "estimate",
                "max_concurrency": est["max_concurrency"],
                "limited_by": est["limited_by"],
                "e2e_p50_s": est["e2e_p50_s"],
                "price_per_hour": g["price_per_hour"],
                **p,
            }
        )
    return out


def _interp_e2e(rows: list[dict[str, Any]], c: float) -> float:
    pts = [(r["concurrency"], r["e2e_p50_s"]) for r in rows if r.get("e2e_p50_s") is not None]
    pts.sort()
    if c <= pts[0][0]:
        return pts[0][1]
    for (c0, v0), (c1, v1) in zip(pts, pts[1:], strict=False):
        if c0 <= c <= c1:
            return v0 + (v1 - v0) * (c - c0) / (c1 - c0)
    return pts[-1][1]


BOUND_LABEL = {
    "measured": "measured",
    "not reached": "not reached",
    "ttft_p95_ms": "first token",
    "itl_p95_ms": "streaming speed",
    "vram": "GPU memory",
    "cap": "model cap",
}


def markdown_table(table: list[dict[str, Any]]) -> str:
    lines = [
        "| GPU | | Max concurrent requests (SLO) | Bound by | Users per GPU | GPUs | $ / month |",
        "|---|---|--:|---|--:|--:|--:|",
    ]
    for r in table:
        cost = f"{r['monthly_cost']:,.0f}" if r.get("monthly_cost") is not None else "n/a"
        gpus = r["gpus"] if r["gpus"] is not None else "n/a"
        at_least = "≥ " if r["limited_by"] == "not reached" else ""
        lines.append(
            f"| {r['gpu']} | {r['kind']} | {at_least}{r['max_concurrency']} | {BOUND_LABEL[r['limited_by']]} | "
            f"{r['users_per_gpu']:.0f} | {gpus} | {cost} |"
        )
    return "\n".join(lines)


DEFAULTS = {"users": 1000, "think_s": 30, "ttft_ms": 1000, "tok_s": 10, "headroom": 0.2}


def build(
    results: str | Path = "results",
    gpus_path: str | Path = "configs/gpus.yaml",
    docs: str | Path = "docs",
    charts: bool = True,
) -> dict[str, Any]:
    results = Path(results)
    summary = load_summary(results / "summary.csv")
    gpus = load_gpus(gpus_path)
    slo = SLO(DEFAULTS["ttft_ms"], DEFAULTS["tok_s"])
    data: dict[str, Any] = {
        "generated": datetime.now(UTC).date().isoformat(),
        "measured_gpu": gpus["measured"],
        "gpus": gpus["gpus"],
        "price_date": gpus["price_date"],
        "price_note": gpus["price_note"],
        "defaults": DEFAULTS,
        "workloads": {},
    }
    md = []
    for key, spec in WORKLOADS.items():
        rows = rows_for(summary, spec["run"])
        if not rows:
            continue
        wl = workload_model(rows, spec["weights_gb"])
        wl["label"] = spec["label"]
        data["workloads"][key] = wl
        table = capacity_table(
            wl, gpus, slo, DEFAULTS["users"], DEFAULTS["think_s"], DEFAULTS["headroom"]
        )
        md.append(f"### {spec['label']}\n\n{markdown_table(table)}\n")

    docs = Path(docs)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (docs / "data.js").write_text(
        "window.CAPPLAN_DATA = " + json.dumps(data) + ";\n", encoding="utf-8"
    )
    (results / "capacity.md").write_text("\n".join(md), encoding="utf-8")
    if charts:
        make_charts(summary, docs / "img", slo)
    return data


def make_charts(summary: list[dict[str, Any]], out: Path, slo: SLO) -> None:
    from .plots import line_chart

    def series(*runs: tuple[str, str]) -> list[dict[str, Any]]:
        return [{"label": lab, "rows": rows_for(summary, run)} for run, lab in runs]

    head = series(("brief-tuned", "brief model"), ("chat-tuned", "chat"))
    if all(s["rows"] for s in head):
        line_chart(
            head,
            "output_tok_per_s",
            out / "throughput",
            "Total output tokens per second",
            "tokens / s (all requests)",
            subtitle="Qwen3-1.7B Q8_0, RTX 5070 Laptop (8 GB), Ollama, slots = requests",
        )
        line_chart(
            head,
            "ttft_p95_ms",
            out / "ttft",
            "Time to first token, 95th percentile",
            "ms",
            subtitle="Waiting time before the answer starts to stream",
            slo=slo.ttft_p95_ms,
            slo_label=f"SLO {slo.ttft_p95_ms:.0f} ms",
        )
        line_chart(
            head,
            "user_tok_per_s_p5",
            out / "user-speed",
            "Streaming speed seen by a user",
            "tokens / s (5th percentile)",
            subtitle="95% of requests stream at least this fast once they have started",
            slo=slo.min_user_tok_per_s,
            slo_label=f"SLO {slo.min_user_tok_per_s:.0f} tok/s",
        )
    slots = series(("brief-tuned", "slots = requests"), ("brief-model", "32 slots"))
    if all(s["rows"] for s in slots):
        line_chart(
            slots,
            "output_tok_per_s",
            out / "slots-throughput",
            "Idle slots cost throughput",
            "tokens / s (all requests)",
            subtitle="Brief model: a server sized to the load vs one with 32 slots",
        )
    quant = series(("q4_K_M", "Q4_K_M"), ("q8_0", "Q8_0"), ("fp16", "FP16"))
    if all(s["rows"] for s in quant):
        line_chart(
            quant,
            "output_tok_per_s",
            out / "quant-throughput",
            "Weight precision vs throughput",
            "tokens / s (all requests)",
            subtitle="Same model, brief workload, 16 slots",
        )
    par = series(("np1", "1 slot"), ("np4", "4 slots"), ("np32", "32 slots"))
    if all(s["rows"] for s in par):
        line_chart(
            par,
            "ttft_p95_ms",
            out / "parallel-ttft",
            "Parallel slots vs time to first token",
            "ms, 95th percentile (log scale)",
            subtitle="With fewer slots than users, requests wait in a queue",
            slo=slo.ttft_p95_ms,
            slo_label=f"SLO {slo.ttft_p95_ms:.0f} ms",
            log_y=True,
        )
