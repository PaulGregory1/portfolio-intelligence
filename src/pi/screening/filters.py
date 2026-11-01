"""Structured screening of the investment universe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pi.analytics.risk import TRADING_DAYS
from pi.data.store import PortfolioData


def instrument_stats(data: PortfolioData) -> pd.DataFrame:
    """Static metadata plus 1-year volatility and return, in the instrument currency."""
    window = data.prices.iloc[-(TRADING_DAYS + 1) :]
    rets = window.pct_change().iloc[1:]
    rows = []
    for ticker, inst in data.instruments.items():
        rows.append(
            {
                "ticker": ticker,
                "name": inst.name,
                "asset_class": inst.asset_class,
                "sub_asset_class": inst.sub_asset_class,
                "region": inst.region,
                "currency": inst.currency,
                "ter_pct": inst.ter_pct,
                "esg_score": inst.esg_score,
                "volatility_1y_pct": float(rets[ticker].std(ddof=1) * np.sqrt(TRADING_DAYS) * 100),
                "return_1y_pct": float(
                    (window[ticker].iloc[-1] / window[ticker].iloc[0] - 1) * 100
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ScreenCriteria:
    asset_class: str | None = None
    sub_asset_class: str | None = None
    region: str | None = None
    currency: str | None = None
    max_ter_pct: float | None = None
    min_esg_score: int | None = None
    max_volatility_pct: float | None = None
    sort_by: str = "ter_pct"


def screen(stats: pd.DataFrame, criteria: ScreenCriteria) -> pd.DataFrame:
    """Apply every non-empty criterion (logical AND). Text criteria ignore case."""
    mask = pd.Series(True, index=stats.index)
    for column in ("asset_class", "sub_asset_class", "region", "currency"):
        wanted = getattr(criteria, column)
        if wanted:
            mask &= stats[column].str.lower() == str(wanted).strip().lower()
    if criteria.max_ter_pct is not None:
        mask &= stats["ter_pct"] <= criteria.max_ter_pct
    if criteria.min_esg_score is not None:
        mask &= stats["esg_score"] >= criteria.min_esg_score
    if criteria.max_volatility_pct is not None:
        mask &= stats["volatility_1y_pct"] <= criteria.max_volatility_pct
    descending = criteria.sort_by in ("esg_score", "return_1y_pct")
    return stats[mask].sort_values([criteria.sort_by, "ticker"], ascending=[not descending, True])
