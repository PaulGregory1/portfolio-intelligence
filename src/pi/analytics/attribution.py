"""Brinson-Fachler performance attribution, linked over multiple days with Carino's method.

For each day and segment i (weights at the start of the day, R_b the total benchmark return):

    allocation  = (w_p,i - w_b,i) * (r_b,i - R_b)
    selection   =  w_b,i          * (r_p,i - r_b,i)
    interaction = (w_p,i - w_b,i) * (r_p,i - r_b,i)

Their sum over segments equals the gross portfolio return minus the benchmark return. Fees
are reported as a separate effect so that the total equals the net TWR minus the benchmark.
Daily effects are scaled by Carino factors so that they add up exactly over the period.

Segments missing from the benchmark (e.g. Japan in a benchmark without Japanese proxies) get
the total benchmark return as their benchmark return, so that holding them shows up as
interaction rather than as a spurious allocation effect. The cash segment gets a zero return.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from pi.analytics.benchmark import benchmark_segments, carino_scaling
from pi.analytics.history import history
from pi.analytics.periods import Period, resolve_period
from pi.analytics.segments import CASH_SEGMENT, portfolio_segments
from pi.data.store import PortfolioData

EFFECTS = ("allocation", "selection", "interaction")


def brinson_fachler(
    w_p: np.ndarray, r_p: np.ndarray, w_b: np.ndarray, r_b: np.ndarray
) -> dict[str, np.ndarray]:
    """Single-period effects per segment. Arrays are aligned on segments (last axis)."""
    r_p = np.where(w_p == 0, r_b, r_p)  # an empty segment has no selection skill
    total_b = np.sum(w_b * r_b, axis=-1, keepdims=True)
    active_w = w_p - w_b
    return {
        "allocation": active_w * (r_b - total_b),
        "selection": w_b * (r_p - r_b),
        "interaction": active_w * (r_p - r_b),
    }


@dataclass(frozen=True)
class AttributionResult:
    client_id: str
    currency: str
    period: Period
    by: str
    segments: pd.DataFrame
    portfolio_return: float  # net TWR
    benchmark_return: float
    active_return: float
    allocation: float
    selection: float
    interaction: float
    fees: float

    @property
    def effects_total(self) -> float:
        return self.allocation + self.selection + self.interaction + self.fees


def attribution(
    data: PortfolioData,
    client_id: str,
    period: str | None = "YTD",
    by: str = "asset_class",
    start_date: date | None = None,
    end_date: date | None = None,
) -> AttributionResult:
    hist = history(data, client_id)
    per = resolve_period(data.dates, period, start_date, end_date)
    port = portfolio_segments(data, hist, by)
    b_weights, b_returns = benchmark_segments(data, client_id, by)

    idx = port.returns.index
    days = idx[(idx > per.start) & (idx <= per.end)]
    r_bench_total = (b_returns.loc[days] * b_weights).sum(axis=1)

    segments = sorted(set(port.weights.columns) | set(b_weights.index))
    w_p = port.weights.reindex(index=days, columns=segments).fillna(0.0)
    r_p = port.returns.reindex(index=days, columns=segments).fillna(0.0)
    w_b = pd.DataFrame(
        np.broadcast_to(b_weights.reindex(segments).fillna(0.0).to_numpy(), w_p.shape),
        index=days,
        columns=segments,
    )
    r_b = b_returns.reindex(index=days, columns=segments)
    for seg in segments:
        if seg not in b_returns:
            fill = 0.0 if seg == CASH_SEGMENT[by] else r_bench_total
            r_b[seg] = fill

    daily = brinson_fachler(w_p.to_numpy(), r_p.to_numpy(), w_b.to_numpy(), r_b.to_numpy())
    r_net = hist.returns.loc[days]
    fee_daily = r_net - (w_p * r_p).sum(axis=1)
    k = carino_scaling(r_net, r_bench_total).to_numpy()[:, None]
    linked = {name: (eff * k).sum(axis=0) for name, eff in daily.items()}
    fees = float((fee_daily.to_numpy() * k[:, 0]).sum())

    table = pd.DataFrame(
        {
            "segment": segments,
            "portfolio_weight": w_p.mean().to_numpy(),
            "benchmark_weight": w_b.iloc[0].to_numpy() if len(days) else 0.0,
            "portfolio_return": (1.0 + r_p).prod().to_numpy() - 1.0,
            "benchmark_return": (1.0 + r_b).prod().to_numpy() - 1.0,
            **{name: linked[name] for name in EFFECTS},
        }
    )
    table["total"] = table[list(EFFECTS)].sum(axis=1)
    port_total = float(np.prod(1.0 + r_net.to_numpy()) - 1.0)
    bench_total = float(np.prod(1.0 + r_bench_total.to_numpy()) - 1.0)
    return AttributionResult(
        client_id=client_id,
        currency=hist.currency,
        period=per,
        by=by,
        segments=table.sort_values("total", key=np.abs, ascending=False).reset_index(drop=True),
        portfolio_return=port_total,
        benchmark_return=bench_total,
        active_return=port_total - bench_total,
        allocation=float(linked["allocation"].sum()),
        selection=float(linked["selection"].sum()),
        interaction=float(linked["interaction"].sum()),
        fees=fees,
    )
