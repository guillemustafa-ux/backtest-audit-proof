"""The report charts as hand-written SVG.

matplotlib output is not byte-stable across versions and the repo promises byte-identical
regeneration, so the charts are plain SVG text: fixed float formatting, no dependency beyond
numpy, one element per line so a diff of a regenerated chart reads like a diff of the report.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

WIDTH, LEFT, RIGHT = 760, 64, 20
CLAIM, CORRECTED = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 of the palette, CVD-safe as a pair
HEAT_LOW, HEAT_HIGH = "#cde2fb", "#0d366b"   # one-hue sequential ramp, light to dark
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MAX_POINTS = 600
EQ_TOP, EQ_BOT, DD_TOP, DD_BOT, EQ_HEIGHT = 36, 276, 300, 440, 484


def _ticks(lo: float, hi: float, target: int) -> tuple[list[float], float]:
    """Round tick values (1, 2, 2.5 or 5 times a power of ten) covering [lo, hi] in about `target` steps."""
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    mag = 10.0 ** math.floor(math.log10(span / target))
    step = min((m * mag for m in (1.0, 2.0, 2.5, 5.0, 10.0)), key=lambda s: abs(span / s - target))
    return [i * step for i in range(math.floor(lo / step), math.ceil(hi / step) + 1)], step


def _label(v: float, step: float, scale: float = 1.0, suffix: str = "") -> str:
    """A tick label with exactly the decimals its step needs: 10 -> "10", 2.5 -> "2.5", -0.2 at scale 100 -> "-20%".
    A step finer than six decimals gets three significant digits instead of failing.
    """
    v, step = v * scale + 0.0, step * scale
    decimals = next((d for d in range(7) if abs(step * 10**d - round(step * 10**d)) < 1e-9), None)
    return f"{v:.{decimals}f}{suffix}" if decimals is not None else f"{v:.3g}{suffix}"


def _log_ticks(lo: float, hi: float) -> list[float]:
    """1, 2 and 5 times a power of ten inside [lo, hi]; when fewer than three fall inside, the nearest values
    below and above are added until there are at least three."""
    decades = range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 2)
    nice = [m * 10.0**k for k in decades for m in (1.0, 2.0, 5.0)]
    below, inside, above = [v for v in nice if v < lo], [v for v in nice if lo <= v <= hi], [v for v in nice if v > hi]
    while len(inside) < 3:
        inside = [below.pop()] + inside + [above.pop(0)]
    return inside


def _scale(lo: float, hi: float, px0: float, px1: float):
    k = (px1 - px0) / (hi - lo)
    return lambda v: px0 + (v - lo) * k


def _log_scale(lo: float, hi: float, px0: float, px1: float):
    linear = _scale(math.log10(lo), math.log10(hi), px0, px1)
    return lambda v: linear(np.log10(v))


def _thin(n: int) -> np.ndarray:
    """Indices to plot: every ceil(n / MAX_POINTS)-th bar plus the last one."""
    idx = np.arange(0, n, math.ceil(n / MAX_POINTS))
    return idx if idx[-1] == n - 1 else np.append(idx, n - 1)


def _svg(width: int, height: int, title: str, body: list[str]) -> str:
    head = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="11" fill="{INK}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{width / 2:.1f}" y="20" text-anchor="middle" font-size="13" font-weight="bold">{escape(title)}</text>',
    ]
    return "\n".join(head + body + ["</svg>", ""])


def _polyline(cls: str, colour: str, xs: np.ndarray, ys: np.ndarray) -> str:
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<polyline class="{cls}" fill="none" stroke="{colour}" stroke-width="1.5" points="{points}"/>'


def equity_drawdown_svg(claim: np.ndarray | pd.Series, corrected: np.ndarray | pd.Series, title: str) -> str:
    """Two stacked panels: equity compounded from 1.0 for both daily series on a log axis, so a claim at 46x and a
    corrected curve at 0.5x are both visible, and the drawdown from the running peak on a linear axis (the peak
    starts at 1.0, as in `stats.max_drawdown`). Round ticks, the final equity in the legend.
    """
    series = [np.asarray(s, dtype=float) for s in (claim, corrected)]
    n = len(series[0])
    assert n >= 2 and all(len(s) == n and np.isfinite(s).all() and (s > -1.0).all() for s in series)  # equity stays > 0
    equity = [np.cumprod(1.0 + s) for s in series]
    drawdown = [e / np.maximum(np.maximum.accumulate(e), 1.0) - 1.0 for e in equity]
    keep = _thin(n)
    x_ticks, x_step = _ticks(0.0, float(n - 1), 5)
    eq_lo, eq_hi = min(e.min() for e in equity), max(e.max() for e in equity)
    eq_ticks = _log_ticks(eq_lo, eq_hi)
    dd_ticks, dd_step = _ticks(min(min(d.min() for d in drawdown), -0.01), 0.0, 3)
    sx = _scale(0.0, x_ticks[-1], LEFT, WIDTH - RIGHT)
    panels = (
        (EQ_TOP, EQ_BOT, eq_ticks, [_label(v, v) for v in eq_ticks], _log_scale(min(eq_lo, eq_ticks[0]), max(eq_hi, eq_ticks[-1]), EQ_BOT, EQ_TOP), equity, "equity (start 1.0, log scale)"),
        (DD_TOP, DD_BOT, dd_ticks, [_label(v, dd_step, 100.0, "%") for v in dd_ticks], _scale(dd_ticks[0], dd_ticks[-1], DD_BOT, DD_TOP), drawdown, "drawdown from peak"),
    )
    body = []
    xs = sx(keep.astype(float))
    for top, bot, ticks, labels, sy, data, name in panels:
        body.append(f'<rect x="{LEFT}" y="{top}" width="{WIDTH - LEFT - RIGHT}" height="{bot - top}" fill="none" stroke="{AXIS}"/>')
        for v, label in zip(ticks, labels):
            y = sy(v)
            body.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" stroke="{GRID}"/>')
            body.append(f'<text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}">{label}</text>')
        for v in x_ticks:
            body.append(f'<line x1="{sx(v):.1f}" y1="{top}" x2="{sx(v):.1f}" y2="{bot}" stroke="{GRID}"/>')
        mid = (top + bot) / 2
        body.append(f'<text transform="rotate(-90 14 {mid:.1f})" x="14" y="{mid:.1f}" text-anchor="middle" fill="{MUTED}">{name}</text>')
        for cls, colour, values in (("claim", CLAIM, data[0]), ("corrected", CORRECTED, data[1])):
            body.append(_polyline(cls, colour, xs, sy(values[keep])))
    for v in x_ticks:
        body.append(f'<text x="{sx(v):.1f}" y="{DD_BOT + 16}" text-anchor="middle" fill="{MUTED}">{_label(v, x_step)}</text>')
    body.append(f'<text x="{(LEFT + WIDTH - RIGHT) / 2:.1f}" y="{DD_BOT + 34}" text-anchor="middle" fill="{MUTED}">bar</text>')
    for i, (name, colour, e) in enumerate((("Claim", CLAIM, equity[0]), ("Corrected", CORRECTED, equity[1]))):
        y = EQ_TOP + 16 + 16 * i
        body.append(f'<line x1="{LEFT + 10}" y1="{y - 4}" x2="{LEFT + 26}" y2="{y - 4}" stroke="{colour}" stroke-width="1.5"/>')
        body.append(f'<text class="legend" x="{LEFT + 32}" y="{y}" font-size="12">{name} {e[-1]:.2f}</text>')
    return _svg(WIDTH, EQ_HEIGHT, title, body)


def _lerp(t: float) -> str:
    lo, hi = (int(HEAT_LOW[i:i + 2], 16) for i in (1, 3, 5)), (int(HEAT_HIGH[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(lo, hi))


def sensitivity_svg(surface: dict[tuple[float, float, float], float], chosen: tuple[float, float, float], title: str) -> str:
    """Heatmap of the Sharpe at the chosen stop: lookback across, target up, one colour ramp from the slice minimum
    to its maximum, every cell labelled, the chosen cell framed.
    """
    stop = chosen[2]
    cells = {(lb, t): float(v) for (lb, t, s), v in surface.items() if s == stop}
    assert cells and all(math.isfinite(v) for v in cells.values())
    lookbacks, targets = sorted({lb for lb, _ in cells}), sorted({t for _, t in cells})
    lo, hi = min(cells.values()), max(cells.values())
    cw, ch, top = (WIDTH - LEFT - RIGHT) / len(lookbacks), 44, 52
    bottom = top + ch * len(targets)
    body = [f'<text x="{WIDTH / 2:.1f}" y="38" text-anchor="middle" fill="{MUTED}">stop = {stop} ATR, naive rules</text>']
    chosen_frame = None
    for col, lb in enumerate(lookbacks):
        x = LEFT + cw * col
        body.append(f'<text x="{x + cw / 2:.1f}" y="{bottom + 18}" text-anchor="middle" fill="{MUTED}">{lb}</text>')
        for row, t in enumerate(targets):
            y = bottom - ch * (row + 1)
            if col == 0:
                body.append(f'<text x="{LEFT - 8}" y="{y + ch / 2 + 4:.1f}" text-anchor="end" fill="{MUTED}">{t}</text>')
            v = cells.get((lb, t))
            if v is None:
                continue
            share = (v - lo) / (hi - lo) if hi > lo else 0.0
            body.append(f'<rect class="cell" x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2:.1f}" height="{ch - 2}" fill="{_lerp(share)}"/>')
            body.append(f'<text class="value" x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" text-anchor="middle" font-size="12" fill="{SURFACE if share > 0.55 else INK}">{v:.2f}</text>')
            if (lb, t, stop) == tuple(chosen):
                chosen_frame = f'<rect class="chosen" x="{x + 1.5:.1f}" y="{y + 1.5:.1f}" width="{cw - 3:.1f}" height="{ch - 3}" fill="none" stroke="{INK}" stroke-width="3"/>'
    if chosen_frame:
        body.append(chosen_frame)
    body.append(f'<text x="{(LEFT + WIDTH - RIGHT) / 2:.1f}" y="{bottom + 36}" text-anchor="middle" fill="{MUTED}">lookback (bars)</text>')
    mid = (top + bottom) / 2
    body.append(f'<text transform="rotate(-90 14 {mid:.1f})" x="14" y="{mid:.1f}" text-anchor="middle" fill="{MUTED}">target (ATR)</text>')
    return _svg(WIDTH, int(bottom + 48), title, body)


def write_charts(result: dict, out_path: str) -> dict[str, str]:
    """Write both charts under charts/ next to the report and return their report-relative posix paths."""
    out, p = Path(out_path), result["params"]
    series = result["risk"]["series"]
    svgs = {
        "equity": equity_drawdown_svg(series["claim"], series["corrected"], f"Equity and drawdown, claim vs corrected: {p.lookback}-bar breakout, target {p.target_atr} ATR, stop {p.stop_atr} ATR"),
        "sensitivity": sensitivity_svg(result["sensitivity"]["surface"], (p.lookback, p.target_atr, p.stop_atr), "Naive Sharpe across the parameter grid"),
    }
    (out.parent / "charts").mkdir(exist_ok=True)
    paths = {}
    for key, svg in svgs.items():
        paths[key] = f"charts/{out.stem}_{key}.svg"
        with open(out.parent / paths[key], "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
    return paths
