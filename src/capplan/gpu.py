"""Background nvidia-smi sampler (VRAM, utilisation, power) for the length of a run."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import time
from typing import Any

QUERY = "memory.used,memory.total,utilization.gpu,power.draw"


def parse_smi_line(line: str) -> dict[str, float] | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        return None

    def num(s: str) -> float | None:
        try:
            return float(s)
        except ValueError:  # "[N/A]" on some laptops
            return None

    mem, total, util, power = (num(p) for p in parts)
    if mem is None:
        return None
    return {"mem_used_mib": mem, "mem_total_mib": total, "util_pct": util, "power_w": power}


async def query_once(gpu_index: int = 0) -> dict[str, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        f"--query-gpu={QUERY}",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return parse_smi_line(out.decode(errors="replace").strip().splitlines()[0] if out else "")


class GpuSampler:
    """`async with GpuSampler() as s:` polls nvidia-smi every `interval` seconds."""

    def __init__(self, interval: float = 1.0, gpu_index: int = 0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.samples: list[dict[str, float]] = []
        self._task: asyncio.Task | None = None
        self._t0 = 0.0

    async def _loop(self) -> None:
        while True:
            s = await query_once(self.gpu_index)
            if s is not None:
                s["t"] = round(time.perf_counter() - self._t0, 2)
                self.samples.append(s)
            await asyncio.sleep(self.interval)

    async def __aenter__(self) -> GpuSampler:
        self._t0 = time.perf_counter()
        if shutil.which("nvidia-smi") is not None:
            self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        mem = [s["mem_used_mib"] for s in self.samples]
        util = [s["util_pct"] for s in self.samples if s.get("util_pct") is not None]
        power = [s["power_w"] for s in self.samples if s.get("power_w") is not None]
        return {
            "vram_peak_mib": max(mem),
            "vram_total_mib": self.samples[0].get("mem_total_mib"),
            "gpu_util_mean_pct": round(sum(util) / len(util), 1) if util else None,
            "power_mean_w": round(sum(power) / len(power), 1) if power else None,
            "gpu_samples": len(self.samples),
        }
