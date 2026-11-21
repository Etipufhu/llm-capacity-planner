import pytest

from capplan.capacity import (
    GPU,
    SLO,
    estimate_gpu,
    fit_curves,
    fit_line,
    kv_bytes_per_token,
    max_concurrency,
    plan,
    users_per_slot,
    vram_concurrency,
)


def row(c, ttft, itl, e2e=2.0, err=0.0):
    return {
        "concurrency": c,
        "ttft_p95_ms": ttft,
        "itl_p95_ms": itl,
        "e2e_p50_s": e2e,
        "error_rate": err,
    }


ROWS = [row(1, 100, 10), row(2, 150, 20), row(4, 300, 40), row(8, 600, 80), row(16, 1400, 160)]


def test_max_concurrency_interpolates_first_violation():
    # ITL limit 100 ms is crossed between c=8 (80) and c=16 (160): 8 + 20/80*8 = 10
    # TTFT limit 1000 ms is crossed at 8 + 400/800*8 = 12; the ITL bound wins
    assert max_concurrency(ROWS, SLO(1000, 10)) == pytest.approx(10.0)


def test_max_concurrency_edges():
    assert max_concurrency(ROWS, SLO(50, 10)) == 0.0  # fails at c=1
    assert max_concurrency(ROWS, SLO(5000, 1)) == 16.0  # never fails
    with_errors = ROWS[:3] + [row(8, 600, 80, err=0.2)]
    assert max_concurrency(with_errors, SLO(5000, 1)) == 4.0


def test_fit_line_exact():
    a, b = fit_line([1, 2, 4, 8], [12, 14, 18, 26])
    assert a == pytest.approx(10) and b == pytest.approx(2)


def test_fit_curves_skips_queued_points():
    fits = fit_curves(ROWS + [row(32, 9000, 170)], max_c=16)
    assert fits["itl_p95_ms"][1] == pytest.approx(10, rel=0.05)


def test_users_per_slot_littles_law():
    # 5 s answer + 25 s reading: one slot serves 6 users
    assert users_per_slot(5, 25) == 6
    with pytest.raises(ValueError):
        users_per_slot(0, 10)


def test_kv_and_vram_bounds():
    kv = kv_bytes_per_token(28, 8, 128, 2)  # Qwen3-1.7B, fp16 cache
    assert kv == 114688
    g = GPU("x", 300, 24)
    n = vram_concurrency(g, weights_gb=1.4, kv_bytes_token=kv, tokens_per_seq=1500)
    assert n == int((24 * 0.9 - 1.4 - 1.0) * 1e9 // (kv * 1500))


def test_estimate_gpu_scales_with_bandwidth():
    fits = fit_curves(ROWS[:4])  # itl = 10 * c roughly, ttft = 75 c + 25
    slow = estimate_gpu(GPU("slow", 384, 8), fits, 384, SLO(1000, 10), vram_cap=999)
    fast = estimate_gpu(GPU("fast", 768, 80), fits, 384, SLO(1000, 10), vram_cap=999)
    assert fast["max_concurrency"] > slow["max_concurrency"] * 1.8
    assert fast["e2e_p50_s"] < slow["e2e_p50_s"]
    capped = estimate_gpu(GPU("fast", 768, 80), fits, 384, SLO(1000, 10), vram_cap=5)
    assert capped["max_concurrency"] == 5 and capped["limited_by"] == "vram"


def test_plan():
    p = plan(users=1000, max_conc=10, e2e_s=5, think_s=25, price_per_hour=1.0, headroom=0.2)
    assert p["users_per_gpu"] == pytest.approx(48)
    assert p["gpus"] == 21
    assert p["monthly_cost"] == pytest.approx(21 * 730)
    assert plan(10, 0, 5, 25, 1.0)["gpus"] is None
