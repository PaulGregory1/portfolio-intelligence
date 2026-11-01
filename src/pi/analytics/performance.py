"""Time-weighted (TWR) and money-weighted (MWR / XIRR) returns."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from pi.analytics.benchmark import benchmark_returns
from pi.analytics.history import as_float, history
from pi.analytics.periods import Period, resolve_period
from pi.data.store import PortfolioData


def chain(returns: pd.Series | np.ndarray) -> float:
    """Compound a series of periodic returns."""
    return float(np.prod(1.0 + np.asarray(returns, dtype=float)) - 1.0)


def annualise(total_return: float, days: int) -> float:
    return (1.0 + total_return) ** (365.0 / days) - 1.0 if days > 0 else 0.0


def xirr(cashflows: Sequence[tuple[date, float]], days_per_period: float = 365.0) -> float:
    """Internal rate of return of dated cash flows, per `days_per_period` (Actual/365 by default).

    Signs are from the investor's point of view: money put in is negative, money received
    (including the final market value) is positive. Solved by bisection, which is slower
    than Newton but cannot diverge. Returns NaN when no root exists.
    """
    if not cashflows:
        return math.nan
    t0 = min(d for d, _ in cashflows)
    times = np.array([(d - t0).days / days_per_period for d, _ in cashflows])
    amounts = np.array([a for _, a in cashflows])

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** times))

    lo, hi = -0.9999, 100.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return math.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9 or hi - lo < 1e-12:
            return mid
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PerformanceResult:
    client_id: str
    currency: str
    period: Period
    twr: float
    twr_annualised: float | None  # only for periods longer than one year
    mwr_period: float  # money-weighted return over the period (IRR, not annualised)
    mwr_annualised: float | None  # only for periods longer than one year
    benchmark_return: float
    active_return: float  # TWR minus benchmark return
    start_value: float
    end_value: float
    net_flows: float
    income: float
    fees: float
    investment_gain: float  # end value - start value - net flows


def period_slice(series: pd.Series, period: Period) -> pd.Series:
    """Daily returns that belong to the period (start excluded, end included)."""
    idx = series.index
    return series[(idx > period.start) & (idx <= period.end)]


def performance(
    data: PortfolioData,
    client_id: str,
    period: str | None = "YTD",
    start_date: date | None = None,
    end_date: date | None = None,
) -> PerformanceResult:
    hist = history(data, client_id)
    per = resolve_period(data.dates, period, start_date, end_date)
    twr = chain(period_slice(hist.returns, per))
    bench = chain(period_slice(benchmark_returns(data, client_id), per))

    start_value = as_float(hist.total.at[per.start])
    end_value = as_float(hist.total.at[per.end])
    flows = period_slice(hist.flows, per)
    cashflows = [(per.start.date(), -start_value)]
    cashflows += [
        (pd.Timestamp(cast(date, d)).date(), -float(f)) for d, f in flows.items() if f != 0
    ]
    cashflows.append((per.end.date(), end_value))
    # Solving the IRR per period length avoids overflowing (1 + r)^(365/days) on short periods.
    mwr = xirr(cashflows, days_per_period=max(per.days, 1))
    net_flows = float(flows.sum())
    return PerformanceResult(
        client_id=client_id,
        currency=hist.currency,
        period=per,
        twr=twr,
        twr_annualised=annualise(twr, per.days) if per.days > 366 else None,
        mwr_period=mwr,
        mwr_annualised=annualise(mwr, per.days) if per.days > 366 else None,
        benchmark_return=bench,
        active_return=twr - bench,
        start_value=start_value,
        end_value=end_value,
        net_flows=net_flows,
        income=float(period_slice(hist.income.sum(axis=1), per).sum()),
        fees=float(period_slice(hist.fees, per).sum()),
        investment_gain=end_value - start_value - net_flows,
    )
