"""Run a benchmark config: for each server setup, a sweep over concurrency levels.

Each (run, concurrency) point is written to results/raw/<run>/c<NNN>.json with
the config, the server details, the GPU samples summary and every request.
results/summary.csv is rebuilt from those files, so it can be regenerated at
any time with `capplan summarize`.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from . import __version__
from .client import run_load
from .gpu import GpuSampler
from .ollama import DEFAULT_HOST, OllamaServer
from .stats import summarize
from .workloads import load_workload

DEFAULTS: dict[str, Any] = {
    "workload": "brief",
    "max_tokens": 320,
    "temperature": 0.7,
    "concurrency": [1, 2, 4, 8, 16, 32],
    "rounds": 4,
    "min_requests": 16,
    "warmup_requests": 2,
    "cooldown_s": 5,
    "timeout_s": 300,
    "stop_error_rate": 0.5,
    "extra_body": {},
    "server": {"engine": "ollama", "env": {}},
}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = deep_merge(DEFAULTS, raw.get("defaults") or {})
    runs = []
    for r in raw["runs"]:
        cfg = deep_merge(defaults, r)
        if "name" not in cfg or "model" not in cfg:
            raise ValueError("every run needs a name and a model")
        runs.append(cfg)
    return runs


def expand_matched(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """With `match_parallel: true`, one server per concurrency level with that many slots.

    The sub-runs share the run name, so their results land in the same
    results/raw/<name>/ folder; each row records its own slot count.
    """
    if not cfg.get("match_parallel"):
        return [cfg]
    subs = []
    for c in cfg["concurrency"]:
        sub = copy.deepcopy(cfg)
        sub["concurrency"] = [c]
        sub["server"].setdefault("env", {})["OLLAMA_NUM_PARALLEL"] = c
        sub["log_name"] = f"{cfg['name']}-np{c}"
        sub["cooldown_s"] = 0
        subs.append(sub)
    return subs


def num_requests(concurrency: int, rounds: int, min_requests: int) -> int:
    """Enough requests that each worker sends `rounds` of them, and at least `min_requests`."""
    return max(concurrency * rounds, min_requests)


def gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


async def _measure(base_url: str, cfg: dict[str, Any], prompts, c: int, n: int, bar):
    async with GpuSampler(interval=1.0) as sampler:
        results, wall = await run_load(
            base_url,
            cfg["model"],
            prompts,
            concurrency=c,
            num_requests=n,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            extra_body=cfg["extra_body"],
            timeout=cfg["timeout_s"],
            on_done=lambda _r: bar.update(1),
        )
    return results, wall, sampler.summary()


def run_config(cfg: dict[str, Any], out_dir: Path, outer: tqdm | None = None) -> None:
    prompts = load_workload(cfg["workload"])
    srv_cfg = cfg["server"]
    server = None
    if srv_cfg["engine"] == "ollama":
        server = OllamaServer(
            exe=srv_cfg.get("exe"),
            host=srv_cfg.get("host", DEFAULT_HOST),
            env=srv_cfg.get("env"),
            log_path=out_dir / "logs" / f"{cfg.get('log_name', cfg['name'])}.log",
        )
        server.start()
        base_url = server.openai_base
    else:
        base_url = srv_cfg["base_url"]

    try:
        if server:
            server.pull(cfg["model"])
        # Warm-up: loads the model. Not recorded. It uses prompts from the end
        # of the set so the measured requests do not hit a cached prompt.
        asyncio.run(
            run_load(
                base_url,
                cfg["model"],
                prompts[-cfg["warmup_requests"] :],
                concurrency=1,
                num_requests=cfg["warmup_requests"],
                max_tokens=cfg["max_tokens"],
                extra_body=cfg["extra_body"],
                timeout=cfg["timeout_s"],
            )
        )
        meta = {
            "run": cfg["name"],
            "group": cfg.get("group"),
            "model": cfg["model"],
            "workload": cfg["workload"],
            "max_tokens": cfg["max_tokens"],
            "temperature": cfg["temperature"],
            "server": srv_cfg,
            "engine_version": server.version() if server else None,
            "loaded_models": server.loaded() if server else None,
            "gpu": gpu_name(),
            "platform": platform.platform(),
            "capplan_version": __version__,
        }
        for c in cfg["concurrency"]:
            n = num_requests(c, cfg["rounds"], cfg["min_requests"])
            with tqdm(total=n, desc=f"{cfg['name']} c={c}", position=1, leave=False) as bar:
                results, wall, gpu = asyncio.run(_measure(base_url, cfg, prompts, c, n, bar))
            summary = summarize(results, wall, c)
            record = {
                "meta": {**meta, "timestamp": datetime.now(UTC).isoformat(timespec="seconds")},
                "summary": summary,
                "gpu": gpu,
                "requests": [r.to_dict() for r in results],
            }
            path = out_dir / "raw" / cfg["name"] / f"c{c:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            if outer is not None:
                outer.update(1)
                outer.set_postfix_str(
                    f"{cfg['name']} c={c}: {summary['output_tok_per_s']} tok/s, "
                    f"TTFT p95 {summary['ttft_p95_ms']} ms"
                )
            if (summary["error_rate"] or 0) >= cfg["stop_error_rate"]:
                tqdm.write(f"{cfg['name']}: error rate {summary['error_rate']} at c={c}, stopping")
                if outer is not None:
                    outer.update(len(cfg["concurrency"]) - cfg["concurrency"].index(c) - 1)
                break
            time.sleep(cfg["cooldown_s"])
    finally:
        if server:
            server.stop()


def run_sweep(config_paths, out_dir: str | Path = "results", only=None) -> None:
    """Run every config file in `config_paths` (one path or a list) in order."""
    out = Path(out_dir)
    if isinstance(config_paths, str | Path):
        config_paths = [config_paths]
    runs = [r for path in config_paths for r in load_config(path)]
    if only:
        runs = [r for r in runs if r["name"] in set(only)]
    runs = [sub for r in runs for sub in expand_matched(r)]
    steps = sum(len(r["concurrency"]) for r in runs)
    with tqdm(total=steps, desc="sweep", position=0) as outer:
        for cfg in runs:
            outer.set_description(f"sweep [{cfg['name']}]")
            run_config(cfg, out, outer)
    write_summary(out)


SUMMARY_META = ["run", "group", "model", "workload", "max_tokens", "gpu"]
SERVER_ENV = {
    "OLLAMA_NUM_PARALLEL": "num_parallel",
    "OLLAMA_CONTEXT_LENGTH": "context_length",
    "OLLAMA_KV_CACHE_TYPE": "kv_cache_type",
    "OLLAMA_FLASH_ATTENTION": "flash_attention",
}


def summary_rows(out_dir: str | Path = "results") -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(out_dir).glob("raw/*/c*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        meta = rec["meta"]
        row = {k: meta.get(k) for k in SUMMARY_META}
        env = (meta.get("server") or {}).get("env") or {}
        for key, col in SERVER_ENV.items():
            row[col] = env.get(key)
        loaded = meta.get("loaded_models") or []
        row["model_vram_mib"] = round(loaded[0]["size_vram"] / 2**20) if loaded else None
        row.update(rec["summary"])
        row.update(rec.get("gpu") or {})
        row["timestamp"] = meta.get("timestamp")
        rows.append(row)
    return rows


def write_summary(out_dir: str | Path = "results") -> Path:
    rows = summary_rows(out_dir)
    path = Path(out_dir) / "summary.csv"
    if not rows:
        return path
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path
