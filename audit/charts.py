"""The report charts as hand-written SVG.

matplotlib output is not byte-stable across versions and the repo promises byte-identical
regeneration, so the charts are plain SVG text: fixed float formatting, no dependency beyond
numpy, one element per line so a diff of a regenerated chart reads like a diff of the report.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

WIDTH, LEFT, RIGHT = 760, 64, 20
CLAIM, CORRECTED = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 of the palette, CVD-safe as a pair
STEP = "#1baf7a"                             # slot 3: with 1 and 2 the only trio that clears every colour-vision check on all pairs
BAND_QUANTILES = (5.0, 95.0)                 # the random-entry envelope: a neutral fill under the curves, not a fourth hue
HEAT_LOW, HEAT_HIGH = "#cde2fb", "#0d366b"   # one-hue sequential ramp, light to dark
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
HALO = f' stroke="{SURFACE}" stroke-width="3" paint-order="stroke"'   # text over curves or the band stays legible
MAX_POINTS = 600                             # at most this many buckets per curve; each draws its first bar and its extremes
EQ_TOP, EQ_BOT, DD_TOP, DD_BOT, EQ_HEIGHT = 66, 306, 330, 470, 514   # two subtitle lines between the title and the first panel


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
    below and above are added until there are at least three; when more than nine do, only the powers of ten stay."""
    decades = range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 2)
    nice = [(m, m * 10.0**k) for k in decades for m in (1.0, 2.0, 5.0)]
    below, inside, above = [t for t in nice if t[1] < lo], [t for t in nice if lo <= t[1] <= hi], [t for t in nice if t[1] > hi]
    while len(inside) < 3:
        inside = [below.pop()] + inside + [above.pop(0)]
    if len(inside) > 9:
        inside = [t for t in inside if t[0] == 1.0]
    return [v for _, v in inside]


def _x_marks(n: int, dates: pd.DatetimeIndex | None) -> tuple[list[tuple[float, str]], str]:
    """(bar position, label) for the x axis and the axis title. Without dates: round bar numbers, none past the
    last bar, titled "bar". With dates: the first bar of every calendar year, every k-th year when more than eight
    would show, labelled with the year and no title, because the labels say what they are.
    """
    if dates is None:
        ticks, step = _ticks(0.0, float(n - 1), 5)
        return [(v, _label(v, step)) for v in ticks if v <= n - 1], "bar"
    if len(dates) != n:
        raise ValueError("dates must have one entry per bar")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted: a year tick marks the first bar of the year")
    years = np.asarray(dates.year)
    starts = [i for i in range(1, n) if years[i] != years[i - 1]]
    if not starts:
        return _x_marks(n, None)  # a sample inside one calendar year has no year start to mark: bars, so the axis is not empty
    every = max(1, math.ceil(len(starts) / 8))
    return [(float(i), str(years[i])) for i in starts[::every]], ""


def _scale(lo: float, hi: float, px0: float, px1: float):
    k = (px1 - px0) / (hi - lo)
    return lambda v: px0 + (v - lo) * k


def _log_scale(lo: float, hi: float, px0: float, px1: float):
    linear = _scale(math.log10(lo), math.log10(hi), px0, px1)
    return lambda v: linear(np.log10(v))


def _thin(curve: np.ndarray) -> np.ndarray:
    """Indices to plot for one curve: the first bar of every bucket of ceil(n / MAX_POINTS) bars, the bars of the
    bucket's minimum and maximum, and the last bar. An every-k-th subsample can skip the bar the report quotes:
    on the synthetic run the claim's maximum drawdown sits on bar 2652 of 3000, which a step of 5 never visits,
    so the drawn curve bottomed at 5.1% against the table's 5.7%.
    """
    n = len(curve)
    step = math.ceil(n / MAX_POINTS)
    keep = {n - 1}
    for start in range(0, n, step):
        segment = curve[start:start + step]
        keep.update((start, start + int(segment.argmin()), start + int(segment.argmax())))
    return np.fromiter(sorted(keep), dtype=int)


def _svg(width: int, height: int, title: str, body: list[str]) -> str:
    head = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="11" fill="{INK}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{width / 2:.1f}" y="20" text-anchor="middle" font-size="13" font-weight="bold">{escape(title)}</text>',
    ]
    return "\n".join(head + body + ["</svg>", ""])


def _polyline(cls: str, colour: str, xs: np.ndarray, ys: np.ndarray, dash: str = "") -> str:
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline class="{cls}" fill="none" stroke="{colour}" stroke-width="1.5"{dash_attr} points="{points}"/>'


def _styles(count: int) -> list[tuple[str, str, str]]:
    """(class, colour, dash) per curve: the first is the claim, the last the corrected run, one in between is an
    intermediate step drawn dashed. A fourth hue would not clear the colour-vision checks on every pair, and in a
    line chart every pair can cross, so the chart takes at most three curves.
    """
    if count < 2 or count > 3:
        raise ValueError("two or three curves: the claim, at most one intermediate step, and the corrected run")
    return [("claim", CLAIM, "")] + [("step", STEP, "6 4")] * (count - 2) + [("corrected", CORRECTED, "")]


def equity_drawdown_svg(curves: list[tuple[str, np.ndarray | pd.Series]], title: str, subtitles: Sequence[str] = (),
                        band: np.ndarray | None = None, band_name: str = "", dates: pd.DatetimeIndex | None = None) -> str:
    """Two stacked panels: equity compounded from 1.0 for every daily series on a log axis, so a claim at 46000x and
    a corrected curve at 0.5x are both visible, and the drawdown from the running peak on a linear axis (the peak
    starts at 1.0, as in `stats.max_drawdown`). `curves` is (legend name, daily returns) in drawing order: the claim
    first, the corrected run last, optionally one intermediate step between them (see `_styles`). Round ticks, the
    final equity in the legend and as a direct label at the end of every equity curve (the step hue sits below 3:1
    on the surface, so its identity cannot rest on colour alone); two end labels overlap only when two finals land
    within about 12 px on the log axis, and the legend repeats every value. Up to two subtitle lines say what each
    curve is, because the chart travels without the report.

    `band`, one row of daily returns per control run, draws the bar-by-bar 5th to 95th percentile of the runs'
    equity as a neutral fill under the curves of the equity panel, with `band_name` in the legend; the drawdown
    panel does not get it. An empty band draws nothing. `dates`, one per bar, labels the x axis with calendar years
    instead of bar numbers; the synthetic run leaves it out because its dates are placeholders.
    """
    if isinstance(subtitles, str) or len(subtitles) > 2:
        raise ValueError("subtitles is a sequence of at most two lines, which is what fits between the title and the first panel")
    styles = _styles(len(curves))
    names, series = [name for name, _ in curves], [np.asarray(s, dtype=float) for _, s in curves]
    n = len(series[0])
    if n < 2 or any(len(s) != n or not np.isfinite(s).all() or (s <= -1.0).any() for s in series):
        raise ValueError("every series needs the same length of at least 2, finite values and every return above -1, so that equity stays positive")
    band = None if band is None or len(band) == 0 else np.asarray(band, dtype=float)
    if band is not None and (band.ndim != 2 or band.shape[1] != n or not np.isfinite(band).all() or (band <= -1.0).any()):
        raise ValueError("band needs one row of daily returns per run, each as long as the curves, finite and above -1")
    equity = [np.cumprod(1.0 + s) for s in series]
    drawdown = [e / np.maximum(np.maximum.accumulate(e), 1.0) - 1.0 for e in equity]
    edges = None if band is None else np.percentile(np.cumprod(1.0 + band, axis=1), BAND_QUANTILES, axis=0)
    x_marks, x_title = _x_marks(n, dates)
    eq_lo, eq_hi = min(e.min() for e in equity), max(e.max() for e in equity)
    if edges is not None:
        eq_lo, eq_hi = min(eq_lo, edges[0].min()), max(eq_hi, edges[1].max())
    eq_ticks = _log_ticks(eq_lo, eq_hi)
    dd_ticks, dd_step = _ticks(min(min(d.min() for d in drawdown), -0.01), 0.0, 3)
    sx = _scale(0.0, float(n - 1), LEFT, WIDTH - RIGHT)
    panels = (
        (EQ_TOP, EQ_BOT, eq_ticks, [_label(v, v) for v in eq_ticks], _log_scale(min(eq_lo, eq_ticks[0]), max(eq_hi, eq_ticks[-1]), EQ_BOT, EQ_TOP), equity, "equity (start 1.0, log scale)"),
        (DD_TOP, DD_BOT, dd_ticks, [_label(v, dd_step, 100.0, "%") for v in dd_ticks], _scale(dd_ticks[0], dd_ticks[-1], DD_BOT, DD_TOP), drawdown, "drawdown from peak"),
    )
    body = [f'<text x="{WIDTH / 2:.1f}" y="{38 + 14 * i}" text-anchor="middle" fill="{MUTED}">{escape(line)}</text>' for i, line in enumerate(subtitles)]
    for top, bot, ticks, labels, sy, data, name in panels:
        body.append(f'<rect x="{LEFT}" y="{top}" width="{WIDTH - LEFT - RIGHT}" height="{bot - top}" fill="none" stroke="{AXIS}"/>')
        if top == EQ_TOP and edges is not None:
            # the envelope goes under the gridlines and the curves: upper edge forward, lower edge back, each thinned
            keep = np.union1d(_thin(edges[1]), _thin(edges[0]))
            xs = sx(keep.astype(float))
            outline = list(zip(xs, sy(edges[1][keep]))) + list(zip(xs[::-1], sy(edges[0][keep])[::-1]))
            body.append(f'<polygon class="band" fill="{AXIS}" fill-opacity="0.45" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in outline)}"/>')
        for v, label in zip(ticks, labels):
            y = sy(v)
            body.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" stroke="{GRID}"/>')
            body.append(f'<text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}">{label}</text>')
        for v, _ in x_marks:
            body.append(f'<line x1="{sx(v):.1f}" y1="{top}" x2="{sx(v):.1f}" y2="{bot}" stroke="{GRID}"/>')
        mid = (top + bot) / 2
        body.append(f'<text transform="rotate(-90 14 {mid:.1f})" x="14" y="{mid:.1f}" text-anchor="middle" fill="{MUTED}">{name}</text>')
        for (cls, colour, dash), values in zip(styles, data):
            keep = _thin(values)
            body.append(_polyline(cls, colour, sx(keep.astype(float)), sy(values[keep]), dash))
    for v, label in x_marks:
        body.append(f'<text x="{sx(v):.1f}" y="{DD_BOT + 16}" text-anchor="middle" fill="{MUTED}">{label}</text>')
    if x_title:
        body.append(f'<text x="{(LEFT + WIDTH - RIGHT) / 2:.1f}" y="{DD_BOT + 34}" text-anchor="middle" fill="{MUTED}">{x_title}</text>')
    eq_sy = panels[0][4]
    for i, (name, (cls, colour, dash), e) in enumerate(zip(names, styles, equity)):
        y = EQ_TOP + 16 + 16 * i
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<line x1="{LEFT + 10}" y1="{y - 4}" x2="{LEFT + 26}" y2="{y - 4}" stroke="{colour}" stroke-width="1.5"{dash_attr}/>')
        body.append(f'<text class="legend" x="{LEFT + 32}" y="{y}" font-size="12"{HALO}>{escape(name)}, final {e[-1]:.2f}x</text>')
        # direct label at the curve's end, in text ink with a surface halo so it stays legible over the other curves;
        # a curve that ends on the frame top gets its label just above the frame instead of over its own last pixels
        label_y = eq_sy(e[-1]) - 4
        label_y = EQ_TOP - 3 if label_y < EQ_TOP + 12 else min(label_y, EQ_BOT - 2)
        body.append(f'<text class="end-label {cls}" x="{WIDTH - RIGHT - 4}" y="{label_y:.1f}" text-anchor="end"{HALO}>{e[-1]:.2f}x</text>')
    if edges is not None:
        y = EQ_TOP + 16 + 16 * len(curves)
        body.append(f'<rect x="{LEFT + 10}" y="{y - 8}" width="16" height="8" fill="{AXIS}" fill-opacity="0.45"/>')
        body.append(f'<text class="legend band" x="{LEFT + 32}" y="{y}" font-size="12"{HALO}>{escape(band_name)}</text>')
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


def write_charts(result: dict, out_path: str, dates: pd.DatetimeIndex | None = None) -> dict[str, str]:
    """Write both charts under charts/ next to the report and return their report-relative posix paths. `dates`
    labels the equity chart's x axis with years; pass it for real data only."""
    out, p = Path(out_path), result["params"]
    series, cost_bps, control = result["risk"]["series"], result["costs"]["cost_bps"], result["control"]
    curves = [("Claim", series["claim"]), ("Look-ahead removed only", series["no_lookahead"]), ("Corrected", series["corrected"])]
    svgs = {
        "equity": equity_drawdown_svg(
            curves,
            f"Equity and drawdown: {p.lookback}-bar breakout, target {p.target_atr} ATR, stop {p.stop_atr} ATR",
            (
                "claim as submitted; dashed: only the look-ahead removed (ambiguous bars still booked to the target, no cost)",
                f"corrected: look-ahead removed, ambiguous bars booked to the stop, {cost_bps:.0f} bps per round trip",
            ),
            band=control["band_series"],
            band_name=f"{control['controls']} matched random-entry runs, same cost: {BAND_QUANTILES[0]:.0f}-{BAND_QUANTILES[1]:.0f}% band",
            dates=dates,
        ),
        "sensitivity": sensitivity_svg(result["sensitivity"]["surface"], (p.lookback, p.target_atr, p.stop_atr), "Naive Sharpe across the parameter grid"),
    }
    (out.parent / "charts").mkdir(exist_ok=True)
    paths = {}
    for key, svg in svgs.items():
        paths[key] = f"charts/{out.stem}_{key}.svg"
        with open(out.parent / paths[key], "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
    return paths
