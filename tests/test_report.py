import csv
import json
from pathlib import Path

from capplan.report import build, load_gpus

ROOT = Path(__file__).resolve().parents[1]


def fake_rows(run, scale=1.0):
    rows = []
    for c in (1, 2, 4, 8, 16, 32):
        rows.append(
            {
                "run": run,
                "model": "m",
                "num_parallel": 32,
                "context_length": 1536,
                "kv_cache_type": "q8_0",
                "concurrency": c,
                "error_rate": 0.0,
                "ttft_p50_ms": 80 * c * scale,
                "ttft_p95_ms": 100 * c * scale,
                "itl_p50_ms": (8 + 2 * c) * scale,
                "itl_p95_ms": (9 + 2.5 * c) * scale,
                "e2e_p50_s": (1 + 0.3 * c) * scale,
                "output_tok_per_s": 110 * c**0.7,
                "user_tok_per_s_p5": 1000 / (9 + 2.5 * c),
                "mean_prompt_tokens": 750,
                "mean_output_tokens": 130,
                "vram_peak_mib": 7000,
            }
        )
    return rows


def test_build_writes_calculator_data_and_charts(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    rows = fake_rows("brief-tuned") + fake_rows("chat-tuned", 0.8)
    with open(results / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    docs = tmp_path / "docs"
    data = build(results, ROOT / "configs" / "gpus.yaml", docs, charts=True)

    assert set(data["workloads"]) == {"brief", "chat"}
    brief = data["workloads"]["brief"]
    assert brief["kv_bytes_token"] == 2 * 28 * 8 * 128 * 34 / 32
    a, b = brief["fits"]["ttft_p95_ms"]
    assert abs(a) < 1e-6 and abs(b - 100) < 1e-6
    assert json.loads((docs / "data.json").read_text())["price_date"]
    assert (docs / "data.js").read_text().startswith("window.CAPPLAN_DATA")
    md = (results / "capacity.md").read_text()
    assert "RTX 5070 Laptop" in md and "H100" in md
    assert (docs / "img" / "throughput-light.png").exists()
    assert (docs / "img" / "ttft-dark.png").exists()


def test_gpu_prices_are_per_gpu():
    gpus = {g["name"]: g for g in load_gpus(ROOT / "configs" / "gpus.yaml")["gpus"]}
    assert gpus["H100 80GB"]["price_per_hour"] == round(55.04 / 8, 4)
    assert gpus["L4"]["price_per_hour"] == 0.8048
