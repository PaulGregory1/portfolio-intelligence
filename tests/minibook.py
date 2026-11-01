"""Tiny hand-made books, small enough to check analytics results with pen and paper."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pi.data.store import PortfolioData
from pi.data.universe import Instrument


def _instrument(ticker: str, asset_class: str, region: str) -> dict[str, Any]:
    inst = Instrument.model_validate(
        {
            "ticker": ticker,
            "name": f"{ticker} fund",
            "asset_class": asset_class,
            "sub_asset_class": "test",
            "region": region,
            "currency": "CHF",
            "ter_pct": 0.1,
            "esg_score": 50,
            "description": "Test.",
            "yf_ticker": None,
            "mu": 0.0,
            "sigma": 0.0,
            "income_yield": 0.0,
        }
    )
    return {**inst.model_dump(exclude={"mu", "sigma"}), "synthetic_prices": True}


def mini_book(
    prices: dict[str, list[float]],
    transactions: list[tuple[int, str, str | None, float, float]],
    start: str = "2026-01-05",
) -> PortfolioData:
    """Build a one-client CHF book.

    `AAA` is an equity fund and `BBB` a bond fund; the client has a 50/50 "balanced" mandate
    whose benchmark holds exactly these two funds. Transactions are
    (day index, type, ticker, quantity, amount in CHF).
    """
    n = len(next(iter(prices.values())))
    dates = pd.bdate_range(start, periods=n)
    price_rows = [
        {"date": d.date(), "ticker": t, "close": p[i]}
        for t, p in prices.items()
        for i, d in enumerate(dates)
    ]
    fx_rows = [
        {"date": d.date(), "currency": c, "chf_per_unit": 1.0}
        for d in dates
        for c in ("CHF", "EUR", "USD")
    ]
    tx_rows = []
    for k, (day, kind, ticker, qty, amount) in enumerate(transactions):
        price = prices[ticker][day] if ticker else None
        tx_rows.append(
            {
                "txn_id": f"T{k:03d}",
                "client_id": "X1",
                "date": dates[day].date(),
                "type": kind,
                "ticker": ticker,
                "quantity": qty,
                "price": price,
                "fx_rate": 1.0 if ticker else None,
                "amount": amount,
                "currency": "CHF",
            }
        )
    tables = {
        "instruments": pd.DataFrame(
            [
                _instrument("AAA", "equity", "Switzerland"),
                _instrument("BBB", "fixed_income", "Europe"),
            ]
        ),
        "prices": pd.DataFrame(price_rows),
        "fx": pd.DataFrame(fx_rows),
        "bankers": pd.DataFrame({"banker_id": ["B9"], "name": ["Test Banker"]}),
        "clients": pd.DataFrame(
            [
                {
                    "client_id": "X1",
                    "name": "Test Client",
                    "banker_id": "B9",
                    "currency": "CHF",
                    "mandate": "balanced",
                    "profile": "test",
                }
            ]
        ),
        "mandates": pd.DataFrame(
            {
                "mandate": ["balanced"] * 2,
                "asset_class": ["equity", "fixed_income"],
                "weight": [0.5, 0.5],
            }
        ),
        "benchmark_proxies": pd.DataFrame(
            {
                "asset_class": ["equity", "fixed_income"],
                "ticker": ["AAA", "BBB"],
                "weight": [1.0, 1.0],
            }
        ),
        "transactions": pd.DataFrame(tx_rows),
        "positions": pd.DataFrame(columns=["client_id", "ticker", "quantity", "as_of"]),
    }
    return PortfolioData(tables)
