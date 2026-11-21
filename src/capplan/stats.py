"""Summary statistics for one load-test run."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (same as numpy's default), q in [0, 100]."""
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _get(r: Any, key: str):
    return r[key] if isinstance(r, dict) else getattr(r, key)


def _ok(r: Any) -> bool:
    return _get(r, "error") is None and _get(r, "ttft") is not None


def summarize(results: Iterable[Any], wall: float, concurrency: int) -> dict[str, Any]:
    """Aggregate per-request results (dicts or RequestResult) into one row.

    Throughput is total output tokens over the wall time of the run. Per-user
    speed is 1 / mean inter-token latency of each request, so it is the rate a
    reader sees once the first token has arrived.
    """
    rs = list(results)
    ok = [r for r in rs if _ok(r)]
    n = len(rs)
    out_tokens = sum(_get(r, "output_tokens") for r in ok)
    in_tokens = sum(_get(r, "prompt_tokens") or 0 for r in ok)
    ttft = [_get(r, "ttft") for r in ok]
    e2e = [_get(r, "e2e") for r in ok]
    itl = [_get(r, "itl_mean") for r in ok if _get(r, "itl_mean")]
    user_tps = [1.0 / x for x in itl]

    def ms(v: float | None) -> float | None:
        return None if v is None else round(v * 1000.0, 1)

    def r2(v: float | None) -> float | None:
        return None if v is None else round(v, 2)

    return {
        "concurrency": concurrency,
        "requests": n,
        "errors": n - len(ok),
        "error_rate": round((n - len(ok)) / n, 4) if n else None,
        "wall_s": round(wall, 2),
        "req_per_s": r2(len(ok) / wall) if wall > 0 else None,
        "output_tok_per_s": r2(out_tokens / wall) if wall > 0 else None,
        "input_tok_per_s": r2(in_tokens / wall) if wall > 0 else None,
        "mean_prompt_tokens": r2(in_tokens / len(ok)) if ok else None,
        "mean_output_tokens": r2(out_tokens / len(ok)) if ok else None,
        "ttft_p50_ms": ms(percentile(ttft, 50)),
        "ttft_p95_ms": ms(percentile(ttft, 95)),
        "ttft_p99_ms": ms(percentile(ttft, 99)),
        "itl_p50_ms": ms(percentile(itl, 50)),
        "itl_p95_ms": ms(percentile(itl, 95)),
        "e2e_p50_s": r2(percentile(e2e, 50)),
        "e2e_p95_s": r2(percentile(e2e, 95)),
        "e2e_p99_s": r2(percentile(e2e, 99)),
        "user_tok_per_s_p50": r2(percentile(user_tps, 50)),
        # 5th percentile: 95% of requests streamed at least this fast
        "user_tok_per_s_p5": r2(percentile(user_tps, 5)),
    }
