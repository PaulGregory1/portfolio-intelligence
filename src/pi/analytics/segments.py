"""Daily weights and returns of portfolio segments (asset classes, regions, ...).

Segment returns are holdings-based and gross of fees. Value moving between segments (trades,
dividends paid into the cash account) is treated as a flow, so only market moves count:

    r_g,t = (V_g,t - V_g,t-1 - NT_g,t) / V_g,t-1

where NT is the net value transferred into the segment. Dividends are therefore credited to the
segment of the instrument that paid them, and the cash account has a zero gross return. Fees
are the only difference between the sum of segment contributions and the net TWR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pi.analytics.history import CASH, PortfolioHistory
from pi.data.store import PortfolioData

SEGMENT_KEYS = ("asset_class", "region")
CASH_SEGMENT = {"asset_class": "cash", "region": "Cash"}


def segment_of(data: PortfolioData, ticker: str, by: str) -> str:
    if by not in SEGMENT_KEYS:
        raise ValueError(f"Unknown segmentation '{by}', expected one of {SEGMENT_KEYS}.")
    if ticker == CASH:
        return CASH_SEGMENT[by]
    return str(getattr(data.instruments[ticker], by))


def _group(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return frame.T.groupby(mapping.__getitem__).sum().T


@dataclass(frozen=True)
class SegmentSeries:
    weights: pd.DataFrame  # date x segment, beginning-of-day weights (from t = 1)
    returns: pd.DataFrame  # date x segment, gross daily returns (from t = 1)


def portfolio_segments(data: PortfolioData, hist: PortfolioHistory, by: str) -> SegmentSeries:
    mapping = {t: segment_of(data, t, by) for t in hist.tickers}
    cash_seg = CASH_SEGMENT[by]
    values = _group(hist.values, mapping) if mapping else pd.DataFrame(index=hist.total.index)
    transfers = _group(hist.trades - hist.income, mapping) if mapping else values * 0.0
    if cash_seg not in values:
        values[cash_seg] = 0.0
        transfers[cash_seg] = 0.0
    values[cash_seg] = values[cash_seg] + hist.cash
    transfers[cash_seg] = transfers[cash_seg] + hist.cash.diff().fillna(0.0)

    prev = values.shift(1)
    raw = (values - prev - transfers) / prev.where(prev > 0)
    returns = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).iloc[1:]
    weights = prev.div(hist.total.shift(1), axis=0).iloc[1:].fillna(0.0)
    return SegmentSeries(weights=weights, returns=returns)
