"""The report charts are hand-written SVG: they must parse, be byte-deterministic and carry the series they claim."""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from audit import charts, report, steps
from audit.data import synthetic_prices
from audit.engine import Params, sharpe
from naive_backtest import GRID

SVG = "{http://www.w3.org/2000/svg}"
CLAIM = np.array([0.01, -0.02, 0.03, 0.0, 0.015] * 10)  # 50 bars, ends above 1.0
CORRECTED = np.array([-0.01, 0.0, -0.005, 0.002, -0.02] * 10)  # 50 bars, ends below 1.0
LOOKBACKS, TARGETS = (5, 10, 20, 40, 60), (0.75, 1.0, 1.5, 2.0, 3.0)
SURFACE = {(lb, t, 1.0): round(0.1 * i - 1.0, 2) for i, (lb, t) in enumerate((lb, t) for lb in LOOKBACKS for t in TARGETS)}
CHOSEN = (20, 1.5, 1.0)
TWO = [("Claim", CLAIM), ("Corrected", CORRECTED)]


def _root(svg: str) -> ET.Element:
    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    return root


def _points(polyline: ET.Element) -> list[tuple[float, float]]:
    return [tuple(float(v) for v in pair.split(",")) for pair in polyline.get("points").split()]


def test_equity_svg_has_two_polylines_per_panel_and_the_final_equity_in_the_legend():
    svg = charts.equity_drawdown_svg(TWO, "test")
    root = _root(svg)
    assert len(list(root.iter(f"{SVG}polyline"))) == 4
    assert f"Claim, final {np.prod(1.0 + CLAIM):.2f}x" in svg and f"Corrected, final {np.prod(1.0 + CORRECTED):.2f}x" in svg


def test_equity_svg_draws_one_intermediate_step_dashed_and_refuses_a_fourth_curve():
    step = np.array([0.002, -0.001, 0.0, 0.001, -0.002] * 10)
    svg = charts.equity_drawdown_svg([("Claim", CLAIM), ("Look-ahead removed only", step), ("Corrected", CORRECTED)], "t")
    root = _root(svg)
    polys = list(root.iter(f"{SVG}polyline"))
    assert [p.get("class") for p in polys] == ["claim", "step", "corrected"] * 2
    assert all((p.get("class") == "step") == bool(p.get("stroke-dasharray")) for p in polys)  # only the step is dashed
    swatches = [l for l in root.iter(f"{SVG}line") if l.get("x1") == str(charts.LEFT + 10)]
    assert [bool(l.get("stroke-dasharray")) for l in swatches] == [False, True, False]  # and so is its legend swatch
    assert f"Look-ahead removed only, final {np.prod(1.0 + step):.2f}x" in svg
    end_labels = [t for t in root.iter(f"{SVG}text") if (t.get("class") or "").startswith("end-label")]
    assert [t.text for t in end_labels] == [f"{np.prod(1.0 + s):.2f}x" for s in (CLAIM, step, CORRECTED)]  # direct labels: the step hue is below 3:1
    assert all(float(t.get("x")) < charts.WIDTH - charts.RIGHT and charts.EQ_TOP - 3 <= float(t.get("y")) < charts.EQ_BOT for t in end_labels)
    # 3.44x and 0.30x: the ticks 0.5, 1, 2 all fit inside, so the axis tops out at the claim and the claim ends on the frame top
    rising, sinking = np.full(50, 0.025), np.full(50, -0.024)
    top_root = _root(charts.equity_drawdown_svg([("Claim", rising), ("Corrected", sinking)], "t"))
    top_labels = [t for t in top_root.iter(f"{SVG}text") if (t.get("class") or "").startswith("end-label")]
    assert top_labels[0].get("y") == str(float(charts.EQ_TOP - 3)) and float(top_labels[1].get("y")) > charts.EQ_TOP + 12  # label above the frame, not over the line
    with pytest.raises(ValueError, match="three"):
        charts.equity_drawdown_svg([("a", CLAIM), ("b", step), ("c", step), ("d", CORRECTED)], "t")


def test_equity_svg_shades_the_control_band_under_the_curves_and_widens_the_axis_to_it():
    # 40 runs of constant daily returns from -3% to +3%: the band spans about 0.26x to 3.7x, well beyond the curves' 0.72x to 1.39x
    band = np.repeat(np.linspace(-0.03, 0.03, 40)[:, None], 50, axis=1)
    svg = charts.equity_drawdown_svg(TWO, "t", band=band, band_name="40 runs, 5-95% band")
    root = _root(svg)
    polygons = list(root.iter(f"{SVG}polygon"))
    assert len(polygons) == 1 and polygons[0].get("class") == "band"
    children = list(root)
    assert children.index(polygons[0]) < min(children.index(p) for p in root.iter(f"{SVG}polyline"))  # drawn under the curves
    ys = [y for _, y in _points(polygons[0])]
    assert min(ys) == charts.EQ_TOP and max(ys) == charts.EQ_BOT  # the axis widened to the band: its edges set both panel limits
    assert all(0.0 <= x <= charts.WIDTH for x, _ in _points(polygons[0]))
    assert "40 runs, 5-95% band" in svg
    assert f"{SVG}polygon" not in {c.tag for c in _root(charts.equity_drawdown_svg(TWO, "t")).iter()}  # no band, no polygon
    assert f"{SVG}polygon" not in {c.tag for c in _root(charts.equity_drawdown_svg(TWO, "t", band=np.zeros((0, 50)))).iter()}  # empty band, nothing drawn
    with pytest.raises(ValueError, match="one row of daily returns per run"):
        charts.equity_drawdown_svg(TWO, "t", band=np.zeros((3, 49)))


def test_equity_svg_escapes_names_and_subtitles_and_caps_the_subtitle_lines():
    svg = charts.equity_drawdown_svg([("P&L <claim>", CLAIM), ("Corrected", CORRECTED)], "t", ("a & b", "c < d"))
    root = _root(svg)  # well-formed despite & and <
    texts = [t.text for t in root.iter(f"{SVG}text")]
    assert f"P&L <claim>, final {np.prod(1.0 + CLAIM):.2f}x" in texts and "a & b" in texts and "c < d" in texts
    for bad in (("1", "2", "3"), "a bare string is not a sequence of lines"):
        with pytest.raises(ValueError, match="at most two lines"):
            charts.equity_drawdown_svg(TWO, "t", bad)


def test_label_handles_fine_steps_without_raising():
    assert charts._label(0.0025, 0.0025) == "0.0025"
    assert charts._label(-0.0025, 0.0025, 100.0, "%") == "-0.25%"
    assert charts._label(1.0, 2.5e-8) == "1"  # finer than six decimals: general format instead of StopIteration


def test_equity_svg_renders_a_narrow_series_with_distinct_tick_labels():
    narrow = np.resize([1e-4, -1e-4], 50)
    root = _root(charts.equity_drawdown_svg([("Claim", narrow), ("Corrected", -narrow)], "t"))
    labels = [t.text for t in root.iter(f"{SVG}text") if t.get("text-anchor") == "end" and t.get("class") is None]  # axis ticks only
    assert len(labels) >= 6 and len(set(labels)) == len(labels)


def test_equity_axis_is_log_with_nice_ticks():
    assert np.allclose(charts._log_ticks(0.45, 32.5), [0.5, 1, 2, 5, 10, 20])
    assert np.allclose(charts._log_ticks(0.99, 1.01), [0.5, 1, 2])  # widened to three ticks
    assert np.allclose(charts._log_ticks(0.43, 46135.8), [1, 10, 100, 1000, 10000])  # more than nine ticks: powers of ten only
    svg = charts.equity_drawdown_svg(TWO, "t")
    root = _root(svg)
    assert "equity (start 1.0, log scale)" in svg and len(list(root.iter(f"{SVG}polyline"))) == 4
    equity_labels = [t.text for t in root.iter(f"{SVG}text") if t.get("text-anchor") == "end" and t.get("class") is None and not t.text.endswith("%")]
    assert equity_labels == ["0.5", "1", "2"]  # the hand-made series stay inside [0.5, 2]


def test_sensitivity_svg_prints_every_cell_and_marks_the_chosen_one():
    svg = charts.sensitivity_svg(SURFACE, CHOSEN, "test")
    root = _root(svg)
    rects = list(root.iter(f"{SVG}rect"))
    cells = [r for r in rects if r.get("class") == "cell"]
    assert len(cells) == 25 and all(r.get("fill", "").startswith("#") for r in cells)
    assert len([r for r in rects if r.get("class") == "chosen"]) == 1
    values = sorted(t.text for t in root.iter(f"{SVG}text") if t.get("class") == "value")
    assert values == sorted(f"{v:.2f}" for v in SURFACE.values())
    assert "stop = 1.0 ATR, naive rules" in svg and "lookback (bars)" in svg and "target (ATR)" in svg


def test_charts_are_deterministic_finite_and_inside_the_viewbox():
    for make in (lambda: charts.equity_drawdown_svg(TWO, "t"), lambda: charts.sensitivity_svg(SURFACE, CHOSEN, "t")):
        a, b = make(), make()
        assert a == b and "nan" not in a and "inf" not in a
        root = _root(a)
        _, _, width, height = (float(v) for v in root.get("viewBox").split())
        for poly in root.iter(f"{SVG}polyline"):
            assert all(0.0 <= x <= width and 0.0 <= y <= height for x, y in _points(poly))


def test_equity_x_axis_ends_at_the_last_bar():
    root = _root(charts.equity_drawdown_svg(TWO, "t"))
    bar_labels = [t.text for t in root.iter(f"{SVG}text") if t.get("y") == str(charts.DD_BOT + 16)]
    assert bar_labels == ["0", "10", "20", "30", "40"]  # 50 bars: no tick at 50, the last bar is 49
    assert all(_points(poly)[-1][0] == charts.WIDTH - charts.RIGHT for poly in root.iter(f"{SVG}polyline"))


def test_equity_svg_rejects_a_return_at_or_below_minus_one():
    with pytest.raises(ValueError, match="-1"):
        charts.equity_drawdown_svg([("Claim", np.array([0.0, -1.0, 0.0])), ("Corrected", np.zeros(3))], "t")


def test_equity_svg_downsamples_long_series():
    rng = np.random.default_rng(0)
    long_a, long_b = rng.normal(0.0, 0.01, 3000), rng.normal(0.0, 0.01, 3000)
    root = _root(charts.equity_drawdown_svg([("Claim", long_a), ("Corrected", long_b)], "t"))
    for poly in root.iter(f"{SVG}polyline"):
        assert poly.get("points").count(",") <= 3 * charts.MAX_POINTS + 1  # per bucket: first bar, minimum, maximum


def test_thinning_keeps_every_bucket_extreme_and_the_drawn_drawdown_bottom():
    # a 3000-bar flat series with one -10% dip: step 5 visits bar 2650 but not 2652, so the dip must be kept by value, not position
    on_grid, off_grid = np.zeros(3000), np.zeros(3000)
    on_grid[2650], on_grid[2651] = -0.1, 1 / 0.9 - 1
    off_grid[2652], off_grid[2653] = -0.1, 1 / 0.9 - 1
    idx = charts._thin(np.cumprod(1.0 + off_grid))
    assert 2652 in idx and 2650 in idx and idx[0] == 0 and idx[-1] == 2999  # the dip is kept; the recovery bar equals its neighbours and is not an extreme
    bottoms = []
    for series in (on_grid, off_grid):
        root = _root(charts.equity_drawdown_svg([("Claim", series), ("Corrected", np.zeros(3000))], "t"))
        claim_drawdown = [p for p in root.iter(f"{SVG}polyline") if p.get("class") == "claim"][1]
        bottoms.append(max(y for _, y in _points(claim_drawdown)))
    assert bottoms[0] == bottoms[1] > charts.DD_TOP + 1  # the dip reaches the same pixel whether or not its bar sits on the sampling grid


def test_risk_profile_exposes_the_daily_series():
    prices = synthetic_prices(400, seed=11)
    r = steps.risk_profile(prices, Params(lookback=5, target_atr=1.0, stop_atr=0.5))
    assert set(r["series"]) == {"claim", "no_lookahead", "corrected"}
    for side in r["series"].values():
        assert isinstance(side, np.ndarray) and len(side) == len(prices)
    # the dashed curve is audit row 1 alone: same Sharpe as the look-ahead step's "after"
    params = Params(lookback=5, target_atr=1.0, stop_atr=0.5)
    assert sharpe(pd.Series(r["series"]["no_lookahead"])) == steps.lookahead(prices, params)["after"]["sharpe"]  # same code path: exact


def test_write_charts_returns_relative_posix_paths(tmp_path):
    result = {"params": Params(*CHOSEN), "costs": {"cost_bps": 20.0}, "risk": {"series": {"claim": CLAIM, "no_lookahead": np.zeros(50), "corrected": CORRECTED}}, "sensitivity": {"surface": SURFACE},
              "control": {"controls": 3, "band_series": np.zeros((3, 50))}}
    paths = charts.write_charts(result, str(tmp_path / "AUDIT_REPORT.md"))
    assert paths == {"equity": "charts/AUDIT_REPORT_equity.svg", "sensitivity": "charts/AUDIT_REPORT_sensitivity.svg"}
    for rel in paths.values():
        _root((tmp_path / rel).read_text(encoding="utf-8"))
    equity = (tmp_path / paths["equity"]).read_text(encoding="utf-8")
    assert "corrected: look-ahead removed, ambiguous bars booked to the stop, 20 bps per round trip" in equity
    assert equity.count('class="step"') == 2 and "Look-ahead removed only, final 1.00x" in equity  # the third curve reaches the file
    assert equity.count('class="band"') == 1 and "3 matched random-entry runs, same cost: 5-95% band" in equity  # and so does the band


def test_report_embeds_the_charts_only_when_given():
    r = report.run_audit(synthetic_prices(500, seed=11), GRID, n_controls=10)
    with_charts = report.render(r, "test", charts={"equity": "charts/x_equity.svg", "sensitivity": "charts/x_sensitivity.svg"})
    assert "## Charts" in with_charts and "](charts/x_equity.svg)" in with_charts and "](charts/x_sensitivity.svg)" in with_charts
    assert with_charts.index("## Risk and exposure") < with_charts.index("## Charts") < with_charts.index("## Material weaknesses")
    assert "## Charts" not in report.render(r, "test")
