"""From measured latency curves to "N users need G GPUs at about $X a month".

Steps
1. SLO check on the measured points: the largest concurrency (requests in
   flight) whose p95 TTFT, p95 inter-token latency and error rate meet the
   SLO. Between two measured points the crossing is found by linear
   interpolation.
2. Little's law: a user who is "active" sends a message, waits W seconds for
   the answer, then reads and types for Z seconds. On average they have
   W / (W + Z) requests in flight, so C slots serve C * (W + Z) / W users.
3. Other GPUs (estimate, not measured): token generation at small batch is
   limited by memory bandwidth, so every latency is divided by
   bandwidth_gpu / bandwidth_measured. The p95 latency curves are fitted as
   a + b * concurrency on the measured points and solved for the SLO; the
   KV cache that fits in VRAM gives a second upper bound.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

HOURS_PER_MONTH = 730


@dataclass
class SLO:
    ttft_p95_ms: float = 1000.0
    min_user_tok_per_s: float = 10.0  # for 95% of requests, i.e. p95 ITL <= 1000 / this
    max_error_rate: float = 0.01

    @property
    def itl_p95_ms(self) -> float:
        return 1000.0 / self.min_user_tok_per_s


def _passes(row: dict[str, Any], slo: SLO) -> bool:
    return (
        row.get("ttft_p95_ms") is not None
        and row.get("itl_p95_ms") is not None
        and row["ttft_p95_ms"] <= slo.ttft_p95_ms
        and row["itl_p95_ms"] <= slo.itl_p95_ms
        and (row.get("error_rate") or 0) <= slo.max_error_rate
    )


def _crossing(c0: float, v0: float, c1: float, v1: float, limit: float) -> float:
    """Concurrency where the line through (c0, v0), (c1, v1) reaches `limit`."""
    if v1 == v0:
        return c1
    return c0 + (limit - v0) * (c1 - c0) / (v1 - v0)


def max_concurrency(rows: Sequence[dict[str, Any]], slo: SLO) -> float:
    """Largest concurrency meeting the SLO, interpolated between measured points.

    Returns 0 if even the lowest measured concurrency fails, and the highest
    measured concurrency if all points pass (the true limit is then higher).
    """
    pts = sorted(rows, key=lambda r: r["concurrency"])
    if not pts or not _passes(pts[0], slo):
        return 0.0
    for lo, hi in zip(pts, pts[1:], strict=False):
        if _passes(hi, slo):
            continue
        c0, c1 = lo["concurrency"], hi["concurrency"]
        if (hi.get("error_rate") or 0) > slo.max_error_rate:
            return float(c0)
        limits = [c1]
        for key, lim in (("ttft_p95_ms", slo.ttft_p95_ms), ("itl_p95_ms", slo.itl_p95_ms)):
            if hi[key] > lim:
                limits.append(_crossing(c0, lo[key], c1, hi[key], lim))
        return max(float(c0), min(limits))
    return float(pts[-1]["concurrency"])


def fit_line(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares a, b for y = a + b * x."""
    n = len(xs)
    if n == 0:
        raise ValueError("no points to fit")
    if n == 1:
        return float(ys[0]), 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return my - b * mx, b


def fit_curves(rows: Sequence[dict[str, Any]], max_c: int | None = None) -> dict[str, list[float]]:
    """Linear fits of p95 TTFT, p95 ITL (ms) and p50 end-to-end time (s) vs concurrency.

    Only error-free points are used, and only points where every request had
    a slot: concurrency up to `max_c` and up to the row's own `num_parallel`
    (past it, requests queue and TTFT jumps).
    """

    def has_slot(r: dict[str, Any]) -> bool:
        n_par = r.get("num_parallel")
        return (max_c is None or r["concurrency"] <= max_c) and (
            n_par is None or r["concurrency"] <= n_par
        )

    pts = [r for r in rows if (r.get("error_rate") or 0) == 0 and has_slot(r)]
    cs = [float(r["concurrency"]) for r in pts]
    return {
        "ttft_p95_ms": list(fit_line(cs, [r["ttft_p95_ms"] for r in pts])),
        "itl_p95_ms": list(fit_line(cs, [r["itl_p95_ms"] for r in pts])),
        "e2e_p50_s": list(fit_line(cs, [r["e2e_p50_s"] for r in pts])),
    }


def users_per_slot(e2e_s: float, think_s: float) -> float:
    """Little's law: active users kept busy by one always-full request slot."""
    if e2e_s <= 0:
        raise ValueError("e2e must be positive")
    return (e2e_s + think_s) / e2e_s


@dataclass
class GPU:
    name: str
    bandwidth_gbs: float
    vram_gb: float
    price_per_hour: float | None = None
    price_source: str | None = None
    measured: bool = False


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per_elem: float) -> float:
    return 2 * layers * kv_heads * head_dim * bytes_per_elem


def vram_concurrency(
    gpu: GPU,
    weights_gb: float,
    kv_bytes_token: float,
    tokens_per_seq: int,
    overhead_gb: float = 1.0,
    usable: float = 0.9,
) -> int:
    """How many sequences' KV caches fit next to the weights."""
    free = gpu.vram_gb * usable - weights_gb - overhead_gb
    if free <= 0:
        return 0
    return int(free * 1e9 // (kv_bytes_token * tokens_per_seq))


def estimate_gpu(
    gpu: GPU,
    fits: dict[str, list[float]],
    ref_bandwidth_gbs: float,
    slo: SLO,
    vram_cap: int,
    c_cap: int = 512,
) -> dict[str, Any]:
    """Max SLO concurrency and per-request time on `gpu` by bandwidth scaling.

    Latency at concurrency c is (a + b * c) / r with r the bandwidth ratio.
    """
    r = gpu.bandwidth_gbs / ref_bandwidth_gbs
    limits = {"vram": float(vram_cap), "cap": float(c_cap)}
    for key, lim in (("ttft_p95_ms", slo.ttft_p95_ms), ("itl_p95_ms", slo.itl_p95_ms)):
        a, b = fits[key]
        if a / r > lim:
            limits[key] = 0.0
        elif b <= 0:
            limits[key] = float(c_cap)
        else:
            limits[key] = (lim * r - a) / b
    c = max(0, math.floor(min(limits.values())))
    binding = min(limits, key=limits.get)
    a, b = fits["e2e_p50_s"]
    e2e = (a + b * max(c, 1)) / r
    return {"ratio": r, "max_concurrency": c, "limited_by": binding, "e2e_p50_s": e2e}


def plan(
    users: int,
    max_conc: float,
    e2e_s: float,
    think_s: float,
    price_per_hour: float | None,
    headroom: float = 0.2,
) -> dict[str, Any]:
    """GPUs and monthly cost for `users` concurrently active chat users."""
    per_gpu = math.floor(max_conc) * users_per_slot(e2e_s, think_s) * (1 - headroom)
    if per_gpu <= 0:
        return {"users_per_gpu": 0.0, "gpus": None, "monthly_cost": None}
    gpus = math.ceil(users / per_gpu)
    cost = gpus * price_per_hour * HOURS_PER_MONTH if price_per_hour is not None else None
    return {"users_per_gpu": per_gpu, "gpus": gpus, "monthly_cost": cost}
