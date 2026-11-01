"""Daily holdings, values and cash flows of a client, rebuilt from the transaction ledger.

Conventions used throughout the analytics:
- all amounts are in the client's reference currency;
- transactions are executed at the close, so external flows happen at the end of the day;
- the daily time-weighted return is r_t = (V_t - F_t) / V_{t-1} - 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat, cast
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd

from pi.data.store import PortfolioData

CASH = "CASH"


def as_float(value: object) -> float:
    """Convert a pandas scalar (typed as a very wide union by pandas-stubs) to float."""
    return float(cast(SupportsFloat, value))


@dataclass(frozen=True)
class PortfolioHistory:
    client_id: str
    currency: str
    quantities: pd.DataFrame  # date x ticker, end-of-day
    cash: pd.Series  # end-of-day cash balance
    values: pd.DataFrame  # date x ticker market value
    total: pd.Series  # total value including cash
    flows: pd.Series  # external flows: deposits (+) and withdrawals (-)
    fees: pd.Series  # fees paid (positive numbers)
    trades: pd.DataFrame  # date x ticker, value bought (+) or sold (-)
    income: pd.DataFrame  # date x ticker, dividends and coupons received
    returns: pd.Series  # daily TWR, first date excluded

    @property
    def tickers(self) -> list[str]:
        return [str(c) for c in self.quantities.columns]


def daily_twr(total: pd.Series, flows: pd.Series) -> pd.Series:
    """Daily time-weighted returns with end-of-day flows."""
    prev = total.shift(1)
    returns = (total - flows) / prev - 1.0
    return returns.iloc[1:].where(prev.iloc[1:] > 0, 0.0)


def _pivot(
    frame: pd.DataFrame, values: pd.Series, dates: pd.DatetimeIndex, cols: list[str]
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(0.0, index=dates, columns=cols)
    table = frame.assign(v=values).pivot_table(
        index="date", columns="ticker", values="v", aggfunc="sum"
    )
    return table.reindex(index=dates, columns=cols).fillna(0.0)


def _by_date(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    return frame.groupby("date")["amount"].sum().reindex(dates).fillna(0.0)


def build_history(data: PortfolioData, client_id: str) -> PortfolioHistory:
    client = data.clients[client_id]
    tx = data.client_transactions(client_id)
    dates = data.dates

    trades = tx[tx["type"].isin(["buy", "sell"])]
    tickers = sorted(str(t) for t in trades["ticker"].unique())
    signed = pd.Series(
        np.where(trades["type"] == "buy", trades["quantity"], -trades["quantity"]),
        index=trades.index,
    )
    quantities = _pivot(trades, signed, dates, tickers).cumsum()
    trade_values = _pivot(trades, -trades["amount"], dates, tickers)
    dividends = tx[tx["type"] == "dividend"]
    income = _pivot(dividends, dividends["amount"], dates, tickers)

    cash = _by_date(tx, dates).cumsum()
    values = quantities * data.prices_in(client.currency)[tickers]
    total = values.sum(axis=1) + cash
    flows = _by_date(tx[tx["type"].isin(["deposit", "withdrawal"])], dates)
    fees = -_by_date(tx[tx["type"] == "fee"], dates)
    return PortfolioHistory(
        client_id=client_id,
        currency=client.currency,
        quantities=quantities,
        cash=cash,
        values=values,
        total=total,
        flows=flows,
        fees=fees,
        trades=trade_values,
        income=income,
        returns=daily_twr(total, flows),
    )


_CACHE: WeakKeyDictionary[PortfolioData, dict[str, PortfolioHistory]] = WeakKeyDictionary()


def history(data: PortfolioData, client_id: str) -> PortfolioHistory:
    """Memoised `build_history` (the data is read-only)."""
    per_data = _CACHE.setdefault(data, {})
    if client_id not in per_data:
        per_data[client_id] = build_history(data, client_id)
    return per_data[client_id]
