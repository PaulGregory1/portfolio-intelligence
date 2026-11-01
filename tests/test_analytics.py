from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pi.analytics.attribution import attribution, brinson_fachler
from pi.analytics.benchmark import benchmark_returns, compare_to_benchmark
from pi.analytics.history import history
from pi.analytics.performance import annualise, chain, performance, xirr
from pi.analytics.periods import resolve_period
from pi.analytics.risk import (
    ex_ante_volatility,
    hhi,
    historical_cvar,
    historical_var,
    information_ratio,
    max_drawdown,
    risk_report,
    tracking_error,
    volatility,
)
from pi.analytics.valuation import exposures, valuation
from pi.data.store import PortfolioData

from .minibook import mini_book

# --- primitives ------------------------------------------------------------------------------


def test_chain_and_annualise() -> None:
    assert chain(np.array([0.10, -0.10])) == pytest.approx(-0.01)
    assert annualise(0.21, 730) == pytest.approx(0.1, abs=1e-3)
    assert annualise(0.05, 0) == 0.0


def test_xirr_simple_cases() -> None:
    assert xirr([(date(2025, 1, 1), -1000.0), (date(2026, 1, 1), 1100.0)]) == pytest.approx(0.10)
    # Two years, no intermediate flow: (1.21)^(1/2) - 1 = 10 %.
    assert xirr([(date(2024, 1, 1), -100.0), (date(2025, 12, 31), 121.0)]) == pytest.approx(
        0.10, abs=1e-3
    )
    assert np.isnan(xirr([(date(2025, 1, 1), 100.0), (date(2026, 1, 1), 100.0)]))
    assert np.isnan(xirr([]))


def test_brinson_fachler_textbook_example() -> None:
    w_p, r_p = np.array([0.6, 0.4]), np.array([0.10, 0.02])
    w_b, r_b = np.array([0.5, 0.5]), np.array([0.08, 0.03])
    eff = brinson_fachler(w_p, r_p, w_b, r_b)
    np.testing.assert_allclose(eff["allocation"], [0.0025, 0.0025])
    np.testing.assert_allclose(eff["selection"], [0.01, -0.005])
    np.testing.assert_allclose(eff["interaction"], [0.002, 0.001])
    total = sum(e.sum() for e in eff.values())
    assert total == pytest.approx(w_p @ r_p - w_b @ r_b)


def test_brinson_fachler_empty_segment_has_no_selection() -> None:
    eff = brinson_fachler(
        np.array([1.0, 0.0]), np.array([0.05, 0.0]), np.array([0.5, 0.5]), np.array([0.04, 0.02])
    )
    assert eff["selection"][1] == 0.0
    assert eff["interaction"][1] == 0.0


def test_risk_primitives() -> None:
    returns = pd.Series([0.10, -0.50, 0.20], index=pd.bdate_range("2026-01-05", periods=3))
    mdd, peak, trough = max_drawdown(returns)
    assert mdd == pytest.approx(0.5)
    assert peak == pd.Timestamp("2026-01-05")
    assert trough == pd.Timestamp("2026-01-06")
    assert max_drawdown(pd.Series(dtype=float))[0] == 0.0

    rng = np.random.default_rng(0)
    r = rng.normal(0.0003, 0.01, 1000)
    assert historical_var(r, 0.95) == pytest.approx(-np.quantile(r, 0.05))
    assert historical_var(r, 0.99) > historical_var(r, 0.95)
    assert historical_cvar(r, 0.95) >= historical_var(r, 0.95)
    assert volatility(r) == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))
    assert volatility(np.full(10, 0.01)) == pytest.approx(0.0)
    assert tracking_error(r, r) == 0.0
    assert information_ratio(r, r) == 0.0
    assert hhi(np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_ex_ante_volatility_single_asset_matches_realised() -> None:
    rng = np.random.default_rng(1)
    rets = pd.DataFrame({"A": rng.normal(0, 0.01, 300), "B": rng.normal(0, 0.02, 300)})
    assert ex_ante_volatility(pd.Series({"A": 1.0}), rets) == pytest.approx(volatility(rets["A"]))
    # Two independent assets: the diversified volatility is below the weighted average.
    mixed = ex_ante_volatility(pd.Series({"A": 0.5, "B": 0.5}), rets)
    assert mixed < 0.5 * volatility(rets["A"]) + 0.5 * volatility(rets["B"])


# --- periods ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2023-10-30", "2026-10-30")


def test_calendar_periods(dates: pd.DatetimeIndex) -> None:
    ytd = resolve_period(dates, "YTD")
    assert (ytd.start, ytd.end) == (pd.Timestamp("2025-12-31"), pd.Timestamp("2026-10-30"))
    q3 = resolve_period(dates, "2026-Q3")
    assert (q3.start, q3.end) == (pd.Timestamp("2026-06-30"), pd.Timestamp("2026-09-30"))
    assert resolve_period(dates, "Q3 2026") == q3
    year = resolve_period(dates, "2025")
    assert (year.start, year.end) == (pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31"))
    month = resolve_period(dates, "2026-02")
    assert (month.start, month.end) == (pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-27"))
    assert resolve_period(dates, "QTD").start == pd.Timestamp("2026-09-30")
    assert resolve_period(dates, "MTD").start == pd.Timestamp("2026-09-30")


def test_trailing_and_custom_periods(dates: pd.DatetimeIndex) -> None:
    one_year = resolve_period(dates, "1Y")
    assert one_year.start == pd.Timestamp("2025-10-30")
    assert resolve_period(dates, "ITD").start == dates[0]
    assert resolve_period(dates, "2023").truncated
    custom = resolve_period(dates, None, date(2026, 3, 1), date(2026, 3, 31))
    assert (custom.start, custom.end) == (pd.Timestamp("2026-02-27"), pd.Timestamp("2026-03-31"))
    with pytest.raises(ValueError, match="Unknown period"):
        resolve_period(dates, "last summer")
    with pytest.raises(ValueError, match="after the last"):
        resolve_period(dates, "2027")


# --- hand-checked mini books -----------------------------------------------------------------


def test_twr_ignores_a_deposit_on_a_constant_return_asset() -> None:
    # Monday to Friday: one return per calendar day, so the daily rate is truly constant.
    prices = [100.0 * 1.01**i for i in range(5)]
    book = mini_book(
        {"AAA": prices, "BBB": [100.0] * 5},
        [
            (0, "deposit", None, 0.0, 1000.0),
            (0, "buy", "AAA", 10.0, -1000.0),
            (2, "deposit", None, 0.0, 5000.0),
            (2, "buy", "AAA", 5000.0 / prices[2], -5000.0),
        ],
    )
    perf = performance(book, "X1", "ITD")
    assert perf.twr == pytest.approx(1.01**4 - 1)
    # With a constant return, money- and time-weighted returns coincide.
    assert perf.mwr_period == pytest.approx(perf.twr, rel=1e-6)
    # The initial deposit is part of the start value (close of day 0), not a period flow.
    assert perf.net_flows == pytest.approx(5000.0)
    assert perf.end_value == pytest.approx(10 * prices[4] + 5000.0 / prices[2] * prices[4])


def test_mwr_rewards_good_timing_but_twr_does_not() -> None:
    # Price falls then recovers; a deposit at the bottom helps the MWR only.
    prices = [100.0, 80.0, 100.0]
    book = mini_book(
        {"AAA": prices, "BBB": [100.0] * 3},
        [
            (0, "deposit", None, 0.0, 1000.0),
            (0, "buy", "AAA", 10.0, -1000.0),
            (1, "deposit", None, 0.0, 800.0),
            (1, "buy", "AAA", 10.0, -800.0),
        ],
    )
    perf = performance(book, "X1", "ITD")
    assert perf.twr == pytest.approx(0.0)
    assert perf.mwr_period > 0.05


def test_attribution_two_assets_by_hand() -> None:
    # Day 1: AAA +10 %, BBB +2 %. Portfolio 60/40, benchmark 50/50.
    book = mini_book(
        {"AAA": [100.0, 110.0], "BBB": [100.0, 102.0]},
        [
            (0, "deposit", None, 0.0, 1000.0),
            (0, "buy", "AAA", 6.0, -600.0),
            (0, "buy", "BBB", 4.0, -400.0),
        ],
    )
    att = attribution(book, "X1", "ITD")
    assert att.portfolio_return == pytest.approx(0.068)
    assert att.benchmark_return == pytest.approx(0.06)
    by_seg = att.segments.set_index("segment")
    # R_b = 6 %; allocation = (0.6-0.5)(0.10-0.06) + (0.4-0.5)(0.02-0.06) = 0.008
    assert att.allocation == pytest.approx(0.008)
    assert att.selection == pytest.approx(0.0, abs=1e-12)
    assert by_seg.loc["equity", "allocation"] == pytest.approx(0.004)
    assert att.effects_total == pytest.approx(att.active_return)


def test_fees_reduce_twr_and_show_as_their_own_effect() -> None:
    book = mini_book(
        {"AAA": [100.0, 100.0, 100.0], "BBB": [100.0] * 3},
        [
            (0, "deposit", None, 0.0, 1000.0),
            (0, "buy", "AAA", 5.0, -500.0),
            (0, "buy", "BBB", 4.0, -400.0),
            (1, "fee", None, 0.0, -10.0),
        ],
    )
    att = attribution(book, "X1", "ITD")
    assert att.portfolio_return == pytest.approx(-0.01)
    assert att.fees == pytest.approx(-0.01)
    assert att.allocation + att.selection + att.interaction == pytest.approx(0.0, abs=1e-12)


def test_dividends_count_in_the_segment_that_paid_them() -> None:
    book = mini_book(
        {"AAA": [100.0, 100.0], "BBB": [100.0, 100.0]},
        [
            (0, "deposit", None, 0.0, 1000.0),
            (0, "buy", "AAA", 10.0, -1000.0),
            (1, "dividend", "AAA", 0.0, 20.0),
        ],
    )
    att = attribution(book, "X1", "ITD")
    by_seg = att.segments.set_index("segment")
    assert by_seg.loc["equity", "portfolio_return"] == pytest.approx(0.02)
    assert by_seg.loc["cash", "portfolio_return"] == pytest.approx(0.0)
    assert att.portfolio_return == pytest.approx(0.02)


# --- invariants on the generated book --------------------------------------------------------

PERIODS = ["MTD", "QTD", "YTD", "1Y", "2025", "2026-Q3", "ITD"]


@pytest.mark.parametrize("period", PERIODS)
def test_attribution_effects_sum_to_active_return(data: PortfolioData, period: str) -> None:
    for client_id in data.clients:
        for by in ("asset_class", "region"):
            att = attribution(data, client_id, period, by=by)
            assert att.effects_total == pytest.approx(att.active_return, abs=1e-10)
            seg_total = att.segments[["allocation", "selection", "interaction"]].to_numpy().sum()
            assert seg_total == pytest.approx(att.allocation + att.selection + att.interaction)


def test_performance_consistency(data: PortfolioData) -> None:
    for client_id in data.clients:
        perf = performance(data, client_id, "1Y")
        att = attribution(data, client_id, "1Y")
        assert perf.twr == pytest.approx(att.portfolio_return)
        assert perf.benchmark_return == pytest.approx(att.benchmark_return)
        bench = benchmark_returns(data, client_id)
        assert len(bench) == len(data.dates) - 1
        assert perf.end_value - perf.start_value == pytest.approx(
            perf.net_flows + perf.investment_gain
        )


def test_valuation_matches_history(data: PortfolioData) -> None:
    for client_id in data.clients:
        val = valuation(data, client_id)
        assert val.total_value == pytest.approx(float(history(data, client_id).total.iloc[-1]))
        assert val.lines["weight"].sum() == pytest.approx(1.0)
        for by in ("asset_class", "region", "currency", "sub_asset_class"):
            assert exposures(val, by).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        exposures(valuation(data, "C01"), "colour")


def test_benchmark_comparison_contributions_add_up(data: PortfolioData) -> None:
    for client_id in data.clients:
        comp = compare_to_benchmark(data, client_id, "YTD")
        assert comp.rows["portfolio_contribution"].sum() + comp.fees == pytest.approx(
            comp.portfolio_return
        )
        assert comp.rows["benchmark_contribution"].sum() == pytest.approx(comp.benchmark_return)
        assert comp.rows["active_weight"].sum() == pytest.approx(0.0, abs=1e-9)


def test_risk_report_is_sane(data: PortfolioData) -> None:
    for client_id in data.clients:
        rep = risk_report(data, client_id, "1Y")
        assert rep.observations > 250
        assert 0 < rep.volatility < 0.5
        assert rep.var_99 >= rep.var_95 > 0
        assert rep.cvar_95 >= rep.var_95
        assert 0 <= rep.max_drawdown < 1
        assert 0 < rep.hhi <= 1
        assert rep.top5_weight <= rep.top10_weight <= 1
        assert sum(rep.currency_exposure.values()) == pytest.approx(1.0)
    concentrated = risk_report(data, "C03")
    diversified = risk_report(data, "C01")
    assert concentrated.volatility > diversified.volatility
