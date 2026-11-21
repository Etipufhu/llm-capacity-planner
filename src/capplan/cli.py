"""Command line entry point: `capplan <command>`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tqdm import tqdm


def cmd_run(args: argparse.Namespace) -> None:
    from .client import run_load
    from .stats import summarize
    from .workloads import load_workload

    prompts = load_workload(args.workload)
    extra = json.loads(args.extra_body) if args.extra_body else None
    with tqdm(total=args.requests, desc=f"c={args.concurrency}") as bar:
        results, wall = asyncio.run(
            run_load(
                args.base_url,
                args.model,
                prompts,
                concurrency=args.concurrency,
                num_requests=args.requests,
                max_tokens=args.max_tokens,
                extra_body=extra,
                on_done=lambda _r: bar.update(1),
            )
        )
    summary = summarize(results, wall, args.concurrency)
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "requests": [r.to_dict() for r in results]}, f, indent=1)


def cmd_sweep(args: argparse.Namespace) -> None:
    from .sweep import run_sweep

    run_sweep(args.config, args.out, only=args.only)


def cmd_summarize(args: argparse.Namespace) -> None:
    from .sweep import write_summary

    print(write_summary(args.out))


def cmd_report(args: argparse.Namespace) -> None:
    from .report import build

    build(args.results, args.gpus, args.docs, charts=not args.no_charts)
    text = Path(args.results, "capacity.md").read_text(encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="capplan", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one load test against a running OpenAI-compatible server")
    r.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    r.add_argument("--model", required=True)
    r.add_argument("--workload", default="brief", help="brief, chat or a .jsonl path")
    r.add_argument("-c", "--concurrency", type=int, default=1)
    r.add_argument("-n", "--requests", type=int, default=16)
    r.add_argument("--max-tokens", type=int, default=320)
    r.add_argument("--extra-body", help="JSON merged into each request, e.g. '{\"top_p\": 0.9}'")
    r.add_argument("--out", help="write summary and per-request results to this JSON file")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="run a benchmark config (starts its own Ollama server)")
    s.add_argument("config", nargs="+", help="one or more config files, run in order")
    s.add_argument("--out", default="results")
    s.add_argument("--only", nargs="*", help="run names to include")
    s.set_defaults(func=cmd_sweep)

    m = sub.add_parser("summarize", help="rebuild results/summary.csv from the raw files")
    m.add_argument("--out", default="results")
    m.set_defaults(func=cmd_summarize)

    rp = sub.add_parser(
        "report", help="capacity table, charts and calculator data from results/summary.csv"
    )
    rp.add_argument("--results", default="results")
    rp.add_argument("--gpus", default="configs/gpus.yaml")
    rp.add_argument("--docs", default="docs")
    rp.add_argument("--no-charts", action="store_true")
    rp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
