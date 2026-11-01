"""Composite benchmarks and the portfolio-versus-benchmark comparison.

A benchmark is a daily-rebalanced mix of investable proxies (see `BENCHMARK_PROXIES`),
unhedged and expressed in the client's reference currency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from pi.analytics.history import as_float, history
from pi.analytics.periods import Period, resolve_period
from pi.analytics.segments import CASH_SEGMENT, portfolio_segments, segment_of
from pi.data.store import PortfolioData


def constituent_returns(data: PortfolioData, currency: str, tickers: list[str]) -> pd.DataFrame:
    return data.prices_in(currency)[tickers].pct_change().iloc[1:]


def benchmark_returns(data: PortfolioData, client_id: str) -> pd.Series:
    """Daily returns of the client's composite benchmark."""
    client = data.clients[client_id]
    weights = data.benchmark_constituents(client.mandate)
    rets = constituent_returns(data, client.currency, list(weights))
    return rets @ pd.Series(weights)


def benchmark_segments(
    data: PortfolioData, client_id: str, by: str
) -> tuple[pd.Series, pd.DataFrame]:
    """Benchmark weight of each segment and the daily return of each segment."""
    client = data.clients[client_id]
    weights = pd.Series(data.benchmark_constituents(client.mandate))
    rets = constituent_returns(data, client.currency, list(weights.index))
    mapping = {str(t): segment_of(data, str(t), by) for t in weights.index}
    seg_weights = weights.groupby(mapping.__getitem__).sum()
    contrib = (rets * weights).T.groupby(mapping.__getitem__).sum().T
    return seg_weights, contrib / seg_weights


def carino_scaling(r_p: pd.Series, r_b: pd.Series) -> pd.Series:
    """Per-day factors k_t / K that link daily arithmetic effects into a period total."""

    def log_ratio(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
        a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        diff = a_arr - b_arr
        same = np.isclose(diff, 0.0, atol=1e-14)
        safe = np.where(same, 1.0, diff)
        return np.where(same, 1.0 / (1.0 + a_arr), (np.log1p(a_arr) - np.log1p(b_arr)) / safe)

    total_p = float(np.prod(1.0 + r_p.to_numpy()) - 1.0)
    total_b = float(np.prod(1.0 + r_b.to_numpy()) - 1.0)
    k_total = float(log_ratio(total_p, total_b))
    return pd.Series(log_ratio(r_p.to_numpy(), r_b.to_numpy()) / k_total, index=r_p.index)


@dataclass(frozen=True)
class BenchmarkComparison:
    client_id: str
    currency: str
    mandate: str
    period: Period
    by: str
    rows: pd.DataFrame  # one row per segment
    portfolio_return: float
    benchmark_return: float
    fees: float  # fee drag, as a (negative) return contribution


def compare_to_benchmark(
    data: PortfolioData,
    client_id: str,
    period: str | None = "YTD",
    by: str = "asset_class",
    start_date: date | None = None,
    end_date: date | None = None,
) -> BenchmarkComparison:
    """Current weights against benchmark weights, plus period returns and contributions."""
    hist = history(data, client_id)
    per = resolve_period(data.dates, period, start_date, end_date)
    port = portfolio_segments(data, hist, by)
    b_weights, b_returns = benchmark_segments(data, client_id, by)

    in_period = (port.returns.index > per.start) & (port.returns.index <= per.end)
    wp, rp = port.weights[in_period], port.returns[in_period]
    rb_seg = b_returns[in_period]
    r_net = hist.returns[in_period]
    r_bench = (rb_seg * b_weights).sum(axis=1)

    k_p = carino_scaling(r_net, pd.Series(0.0, index=r_net.index))
    k_b = carino_scaling(r_bench, pd.Series(0.0, index=r_bench.index))
    contrib_p = (wp * rp).mul(k_p, axis=0).sum()
    contrib_b = (rb_seg * b_weights).mul(k_b, axis=0).sum()
    r_gross = (wp * rp).sum(axis=1)
    fee_drag = float(((r_net - r_gross) * k_p).sum())

    # Weights at the close of the end date.
    total_end = as_float(hist.total.loc[per.end])
    mapping = {t: segment_of(data, t, by) for t in hist.tickers}
    w_now = hist.values.loc[per.end].groupby(mapping.__getitem__).sum() / total_end
    cash_seg = CASH_SEGMENT[by]
    w_now[cash_seg] = (
        as_float(w_now.get(cash_seg, 0.0)) + as_float(hist.cash.loc[per.end]) / total_end
    )

    segments = sorted(set(w_now.index) | set(b_weights.index))
    rows = []
    for seg in segments:
        wp_now = float(w_now.get(seg, 0.0))
        wb = float(b_weights.get(seg, 0.0))
        rows.append(
            {
                "segment": seg,
                "portfolio_weight": wp_now,
                "benchmark_weight": wb,
                "active_weight": wp_now - wb,
                "portfolio_return": float(np.prod(1.0 + rp[seg]) - 1.0) if seg in rp else 0.0,
                "benchmark_return": float(np.prod(1.0 + rb_seg[seg]) - 1.0)
                if seg in rb_seg
                else None,
                "portfolio_contribution": float(contrib_p.get(seg, 0.0)),
                "benchmark_contribution": float(contrib_b.get(seg, 0.0)),
            }
        )
    return BenchmarkComparison(
        client_id=client_id,
        currency=hist.currency,
        mandate=data.clients[client_id].mandate,
        period=per,
        by=by,
        rows=pd.DataFrame(rows).sort_values("portfolio_weight", ascending=False),
        portfolio_return=float(np.prod(1.0 + r_net) - 1.0),
        benchmark_return=float(np.prod(1.0 + r_bench) - 1.0),
        fees=fee_drag,
    )
