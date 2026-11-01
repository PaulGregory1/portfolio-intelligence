"""Market value of each line and aggregated exposures."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pi.analytics.history import CASH, as_float, history
from pi.data.store import PortfolioData

GROUPINGS = ("asset_class", "sub_asset_class", "region", "currency")


@dataclass(frozen=True)
class Valuation:
    client_id: str
    currency: str
    as_of: pd.Timestamp
    total_value: float
    lines: pd.DataFrame  # one row per position plus a CASH row, sorted by weight


def valuation(data: PortfolioData, client_id: str, as_of: pd.Timestamp | None = None) -> Valuation:
    hist = history(data, client_id)
    day = pd.Timestamp(as_of) if as_of is not None else data.end_date
    day = hist.total.index[hist.total.index <= day][-1]
    ref = hist.currency
    rows: list[dict[str, object]] = []
    for ticker in hist.tickers:
        qty = as_float(hist.quantities.at[day, ticker])
        if qty == 0:
            continue
        inst = data.instruments[ticker]
        price = as_float(data.prices.at[day, ticker])
        rate = as_float(data.fx.at[day, inst.currency]) / as_float(data.fx.at[day, ref])
        rows.append(
            {
                "ticker": ticker,
                "name": inst.name,
                "asset_class": inst.asset_class,
                "sub_asset_class": inst.sub_asset_class,
                "region": inst.region,
                "currency": inst.currency,
                "quantity": qty,
                "price": price,
                "fx_rate": rate,
                "market_value_local": qty * price,
                "market_value": qty * price * rate,
            }
        )
    cash = as_float(hist.cash.at[day])
    rows.append(
        {
            "ticker": CASH,
            "name": f"Cash account {ref}",
            "asset_class": "cash",
            "sub_asset_class": "cash",
            "region": "Cash",
            "currency": ref,
            "quantity": cash,
            "price": 1.0,
            "fx_rate": 1.0,
            "market_value_local": cash,
            "market_value": cash,
        }
    )
    lines = pd.DataFrame(rows)
    total = float(lines["market_value"].sum())
    lines["weight"] = lines["market_value"] / total
    lines = lines.sort_values("weight", ascending=False).reset_index(drop=True)
    return Valuation(client_id, ref, day, total, lines)


def exposures(val: Valuation, by: str) -> pd.Series:
    """Weights aggregated by `asset_class`, `sub_asset_class`, `region` or `currency`."""
    if by not in GROUPINGS:
        raise ValueError(f"Unknown grouping '{by}', expected one of {GROUPINGS}.")
    return val.lines.groupby(by)["weight"].sum().sort_values(ascending=False)
