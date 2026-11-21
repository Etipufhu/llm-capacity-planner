"""Charts for the README, each rendered once for a light and once for a dark page."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


def _axes(plt, theme: dict[str, Any], title: str, subtitle: str | None, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=theme["ink2"], fontsize=9)
    fig.text(0.015, 0.95, title, color=theme["ink"], fontsize=12, weight="bold", va="top")
    if subtitle:
        fig.text(0.015, 0.885, subtitle, color=theme["ink2"], fontsize=9, va="top")
    return fig, ax


def _end_labels(ax, ends, theme: dict[str, Any], dpi: float, min_gap_px: float = 22) -> None:
    """Direct labels at the line ends, pushed apart vertically so they never overlap."""
    if not ends:
        return
    ax.get_ylim()  # settle autoscaling before converting to pixels
    placed = sorted(
        ((ax.transData.transform((x, y))[1], x, y, lab) for x, y, lab in ends),
        key=lambda t: t[0],
    )
    ys_px = [p[0] for p in placed]
    for i in range(1, len(ys_px)):
        ys_px[i] = max(ys_px[i], ys_px[i - 1] + min_gap_px)
    for (orig, x, y, lab), new in zip(placed, ys_px, strict=True):
        ax.annotate(
            lab,
            (x, y),
            xytext=(6, (new - orig) * 72 / dpi),
            textcoords="offset points",
            color=theme["ink2"],
            fontsize=8,
            va="center",
        )


def line_chart(
    series: Sequence[dict[str, Any]],
    metric: str,
    out_stem: str | Path,
    title: str,
    ylabel: str,
    subtitle: str | None = None,
    slo: float | None = None,
    slo_label: str | None = None,
    log_y: bool = False,
) -> list[Path]:
    """One line per series (dict with `label` and `rows`) of `metric` vs concurrency.

    Writes <out_stem>-light.png and <out_stem>-dark.png.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(series) > 3:
        raise ValueError("at most three series per chart (colour-blind-safe limit)")
    paths = []
    for mode, theme in THEMES.items():
        fig, ax = _axes(plt, theme, title, subtitle, ylabel)
        xs_all: set[int] = set()
        ends: list[tuple[float, float, str]] = []
        for s, color in zip(series, theme["series"], strict=False):
            pts = sorted(
                (r for r in s["rows"] if r.get(metric) is not None),
                key=lambda r: r["concurrency"],
            )
            xs = [r["concurrency"] for r in pts]
            ys = [r[metric] for r in pts]
            xs_all.update(xs)
            ax.plot(xs, ys, color=color, linewidth=2, label=s["label"], solid_capstyle="round")
            ax.scatter(
                xs, ys, s=28, color=color, edgecolors=theme["surface"], linewidths=1.5, zorder=3
            )
            if pts and len(series) > 1:
                ends.append((xs[-1], ys[-1], s["label"]))
        if slo is not None:
            ax.axhline(slo, color=theme["muted"], linewidth=1, linestyle=(0, (4, 3)))
            if slo_label:
                ax.text(
                    min(xs_all) if xs_all else 1,
                    slo,
                    f" {slo_label}",
                    color=theme["muted"],
                    fontsize=8,
                    va="bottom",
                )
        ax.set_xscale("log", base=2)
        ticks = sorted(xs_all)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("concurrent requests", color=theme["ink2"], fontsize=9)
        if log_y:
            from matplotlib.ticker import FuncFormatter

            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        else:
            ax.set_ylim(bottom=0)
        if xs_all:
            ax.set_xlim(min(xs_all) / 1.25, max(xs_all) * (1.9 if len(series) > 1 else 1.25))
        _end_labels(ax, ends, theme, fig.dpi)
        if len(series) > 1:
            leg = ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=theme["ink2"])
            leg.set_zorder(5)
        fig.subplots_adjust(top=0.8 if subtitle else 0.86, left=0.1, right=0.97, bottom=0.14)
        path = Path(f"{out_stem}-{mode}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=theme["surface"])
        plt.close(fig)
        paths.append(path)
    return paths


def bar_chart(
    items: Sequence[tuple[str, float]],
    out_stem: str | Path,
    title: str,
    xlabel: str,
    subtitle: str | None = None,
    fmt: str = "{:,.0f}",
) -> list[Path]:
    """Horizontal bars, one colour (a single measure), value labels at the bar ends."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for mode, theme in THEMES.items():
        fig, ax = _axes(plt, theme, title, subtitle, "")
        labels = [k for k, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        ax.barh(labels, vals, color=theme["series"][0], height=0.6)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", colors=theme["ink2"], labelsize=9)
        top = max(vals) if vals else 1
        for y, v in enumerate(vals):
            ax.text(v + top * 0.01, y, fmt.format(v), color=theme["ink2"], fontsize=8, va="center")
        ax.set_xlim(0, top * 1.15)
        ax.set_xlabel(xlabel, color=theme["ink2"], fontsize=9)
        fig.subplots_adjust(top=0.8 if subtitle else 0.86, left=0.22, right=0.97, bottom=0.14)
        path = Path(f"{out_stem}-{mode}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=theme["surface"])
        plt.close(fig)
        paths.append(path)
    return paths
