"""Synthetic private-bank book: bankers, clients, mandates, benchmarks and 3 years of transactions.

Everything is deterministic given the seed. Positions are never written directly: they are
the result of replaying the simulated transactions, and are stored only as a snapshot that
the tests reconcile against the transaction ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pi.data.fx import load_fx
from pi.data.prices import business_days, load_prices
from pi.data.universe import INSTRUMENTS, UNIVERSE, AssetClass, Currency

ASSET_CLASSES: tuple[AssetClass, ...] = ("equity", "fixed_income", "alternatives", "cash")

# Strategic asset allocation of each mandate (cash has a zero benchmark weight).
MANDATES: dict[str, dict[AssetClass, float]] = {
    "conservative": {"equity": 0.20, "fixed_income": 0.70, "alternatives": 0.10, "cash": 0.0},
    "balanced": {"equity": 0.50, "fixed_income": 0.40, "alternatives": 0.10, "cash": 0.0},
    "growth": {"equity": 0.75, "fixed_income": 0.15, "alternatives": 0.10, "cash": 0.0},
}

# Each asset-class sleeve of the composite benchmarks is represented by investable proxies.
BENCHMARK_PROXIES: dict[AssetClass, dict[str, float]] = {
    "equity": {"CHSPI": 0.30, "EXSA": 0.20, "VTI": 0.40, "EEM": 0.10},
    "fixed_income": {"CSBGC7": 0.25, "CHCORP": 0.25, "IEGA": 0.25, "AGG": 0.25},
    "alternatives": {"ZGLD": 0.50, "SRECHA": 0.50},
    "cash": {"CHFMM": 1.0},
}

BANKERS = {"B1": "Anna Keller", "B2": "Marc Dubois", "B3": "Luca Rossi"}

MANAGEMENT_FEE = 0.008  # per year, charged quarterly on the portfolio value
CASH_BUFFER = 0.015  # cash kept aside when investing
REBALANCE_EVERY = 126  # business days


@dataclass(frozen=True)
class ClientSpec:
    client_id: str
    name: str
    banker_id: str
    currency: Currency
    mandate: str
    # Actual asset-class allocation, deliberately away from the mandate.
    allocation: dict[AssetClass, float]
    # Relative weights inside each asset class.
    holdings: dict[AssetClass, dict[str, float]]
    profile: str


CLIENTS: tuple[ClientSpec, ...] = (
    ClientSpec(
        "C01",
        "Elena Brunner",
        "B1",
        "CHF",
        "conservative",
        {"equity": 0.27, "fixed_income": 0.62, "alternatives": 0.11},
        {
            "equity": {"CHSPI": 4, "NESN": 2, "NOVN": 2, "VTI": 2},
            "fixed_income": {"CSBGC7": 3, "CHCORP": 3, "CSBGC0": 2, "AGG": 1},
            "alternatives": {"ZGLD": 1, "SRECHA": 1},
        },
        "Retired, strong Swiss home bias, slightly overweight equities.",
    ),
    ClientSpec(
        "C02",
        "Thomas Weber",
        "B1",
        "EUR",
        "balanced",
        {"equity": 0.52, "fixed_income": 0.36, "alternatives": 0.12},
        {
            "equity": {"EXSA": 3, "CSX5": 2, "SAP": 1, "MC": 1, "ASML": 1, "SPY": 2},
            "fixed_income": {"IEGA": 3, "IEAC": 3, "IHYG": 1},
            "alternatives": {"IPRP": 1, "GLD": 1},
        },
        "Entrepreneur based in Munich, overweight European equity and EUR credit.",
    ),
    ClientSpec(
        "C03",
        "Hoffmann Family Office",
        "B1",
        "USD",
        "growth",
        {"equity": 0.84, "fixed_income": 0.10, "alternatives": 0.06},
        {
            "equity": {"QQQ": 3, "AAPL": 2.5, "MSFT": 2.5, "SPY": 2, "ASML": 1, "EEM": 0.5},
            "fixed_income": {"IEF": 1, "LQD": 1},
            "alternatives": {"GLD": 1},
        },
        "Tech-concentrated growth portfolio with large single-stock positions.",
    ),
    ClientSpec(
        "C04",
        "Sophie Laurent",
        "B2",
        "CHF",
        "balanced",
        {"equity": 0.44, "fixed_income": 0.36, "alternatives": 0.20},
        {
            "equity": {"CSSMI": 3, "URTH": 3, "EWJ": 1},
            "fixed_income": {"CSBGC7": 2, "CHCORP": 2, "IEGA": 1, "TIP": 1},
            "alternatives": {"ZGLD": 3, "SRECHA": 1},
        },
        "Cautious investor with a large gold allocation as a crisis hedge.",
    ),
    ClientSpec(
        "C05",
        "Giovanni Bernasconi",
        "B2",
        "CHF",
        "growth",
        {"equity": 0.78, "fixed_income": 0.12, "alternatives": 0.10},
        {
            "equity": {"NESN": 3, "NOVN": 3, "ROG": 3, "UBSG": 2, "CSSMIM": 2, "SPY": 2},
            "fixed_income": {"CHCORP": 1, "CSBGC0": 1},
            "alternatives": {"SRECHA": 1},
        },
        "Ticino entrepreneur concentrated in Swiss blue chips.",
    ),
    ClientSpec(
        "C06",
        "Clara Meier",
        "B2",
        "EUR",
        "conservative",
        {"equity": 0.18, "fixed_income": 0.74, "alternatives": 0.08},
        {
            "equity": {"EXSA": 2, "URTH": 1},
            "fixed_income": {"IEGA": 2, "IEAC": 2, "IHYG": 2, "HYG": 1.5, "EMB": 1},
            "alternatives": {"IPRP": 1},
        },
        "Income-oriented conservative investor reaching for yield in high yield bonds.",
    ),
    ClientSpec(
        "C07",
        "Olivier Favre",
        "B3",
        "USD",
        "balanced",
        {"equity": 0.53, "fixed_income": 0.38, "alternatives": 0.09},
        {
            "equity": {"VTI": 3, "EEM": 2, "VWO": 2, "EWJ": 1, "JNJ": 1},
            "fixed_income": {"AGG": 2, "EMB": 2, "TIP": 1, "LQD": 1},
            "alternatives": {"VNQ": 1, "GLD": 1},
        },
        "Geneva-based expatriate with an emerging-markets tilt in equity and debt.",
    ),
    ClientSpec(
        "C08",
        "Nina Schmid",
        "B3",
        "CHF",
        "growth",
        {"equity": 0.80, "fixed_income": 0.08, "alternatives": 0.12},
        {
            "equity": {"SUSW": 5, "CHSPI": 2, "MSFT": 1, "SAP": 1},
            "fixed_income": {"CHCORP": 1, "CSBGC7": 0.5},
            "alternatives": {"ZGLD": 1, "SRECHA": 1},
        },
        "Young professional with a sustainability (ESG) focus, underweight bonds.",
    ),
)


@dataclass
class _Book:
    """Running state of one client while the transactions are simulated."""

    spec: ClientSpec
    targets: dict[str, float]
    cash: float = 0.0
    qty: dict[str, float] = field(default_factory=dict)
    rows: list[dict[str, object]] = field(default_factory=list)


@dataclass
class GeneratedData:
    instruments: pd.DataFrame
    prices: pd.DataFrame  # long format: date, ticker, close
    fx: pd.DataFrame  # long format: date, currency, chf_per_unit
    bankers: pd.DataFrame
    clients: pd.DataFrame
    mandates: pd.DataFrame
    benchmark_proxies: pd.DataFrame
    transactions: pd.DataFrame
    positions: pd.DataFrame
    synthetic_tickers: list[str]

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "instruments": self.instruments,
            "prices": self.prices,
            "fx": self.fx,
            "bankers": self.bankers,
            "clients": self.clients,
            "mandates": self.mandates,
            "benchmark_proxies": self.benchmark_proxies,
            "transactions": self.transactions,
            "positions": self.positions,
        }


def _targets(spec: ClientSpec, rng: np.random.Generator) -> dict[str, float]:
    """Line-level target weights, with a small random perturbation inside each sleeve."""
    targets: dict[str, float] = {}
    invested = 1.0 - CASH_BUFFER
    for asset_class, rel in spec.holdings.items():
        noisy = {t: w * rng.uniform(0.85, 1.15) for t, w in rel.items()}
        total = sum(noisy.values())
        for ticker, w in noisy.items():
            targets[ticker] = invested * spec.allocation[asset_class] * w / total
    return targets


class _Simulator:
    def __init__(self, prices: pd.DataFrame, fx: pd.DataFrame) -> None:
        self.dates = pd.DatetimeIndex(prices.index)
        self.prices = {str(t): prices[t].to_numpy(dtype=float) for t in prices.columns}
        self.fx = {str(c): fx[c].to_numpy(dtype=float) for c in fx.columns}
        self.i = 0  # index of the current business day

    # --- market helpers -----------------------------------------------------------------------
    def price(self, ticker: str) -> float:
        return float(self.prices[ticker][self.i])

    def rate(self, ticker: str, ref: str) -> float:
        ccy = INSTRUMENTS[ticker].currency
        return float(self.fx[ccy][self.i] / self.fx[ref][self.i])

    def value(self, book: _Book) -> float:
        ref = book.spec.currency
        held = sum(q * self.price(t) * self.rate(t, ref) for t, q in book.qty.items())
        return held + book.cash

    # --- ledger -------------------------------------------------------------------------------
    def record(
        self, book: _Book, kind: str, amount: float, ticker: str | None = None, quantity: float = 0
    ) -> None:
        amount = round(amount, 2)
        book.cash = round(book.cash + amount, 2)
        price = self.price(ticker) if ticker else None
        rate = self.rate(ticker, book.spec.currency) if ticker else None
        book.rows.append(
            {
                "client_id": book.spec.client_id,
                "date": self.dates[self.i].date(),
                "type": kind,
                "ticker": ticker,
                "quantity": float(quantity),
                "price": price,
                "fx_rate": rate,
                "amount": amount,
                "currency": book.spec.currency,
            }
        )

    def trade(self, book: _Book, ticker: str, quantity: float) -> None:
        if quantity == 0:
            return
        gross = quantity * self.price(ticker) * self.rate(ticker, book.spec.currency)
        self.record(book, "buy" if quantity > 0 else "sell", -gross, ticker, abs(quantity))
        book.qty[ticker] = book.qty.get(ticker, 0.0) + quantity
        if book.qty[ticker] == 0:
            del book.qty[ticker]

    def shares_for(self, book: _Book, ticker: str, amount: float) -> float:
        unit = self.price(ticker) * self.rate(ticker, book.spec.currency)
        return float(np.floor(amount / unit))

    # --- portfolio actions --------------------------------------------------------------------
    def rebalance(self, book: _Book, threshold: float = 0.0) -> None:
        """Trade every line back to its target weight; sells first so cash stays positive."""
        total = self.value(book)
        orders: dict[str, float] = {}
        for ticker, target in book.targets.items():
            current = (
                book.qty.get(ticker, 0.0)
                * self.price(ticker)
                * self.rate(ticker, book.spec.currency)
            )
            if abs(current / total - target) <= threshold:
                continue
            delta = self.shares_for(book, ticker, target * total - current)
            if delta < 0 and -delta > book.qty.get(ticker, 0.0):
                delta = -book.qty.get(ticker, 0.0)
            orders[ticker] = delta
        for ticker, delta in sorted(orders.items(), key=lambda kv: kv[1]):
            if delta > 0:
                cost = delta * self.price(ticker) * self.rate(ticker, book.spec.currency)
                if cost > book.cash:
                    delta = self.shares_for(book, ticker, book.cash)
            self.trade(book, ticker, delta)

    def raise_cash(self, book: _Book, amount: float) -> None:
        """Sell pro rata so that at least `amount` of cash is available."""
        missing = amount - book.cash
        if missing <= 0:
            return
        invested = self.value(book) - book.cash
        for ticker in sorted(book.qty):
            unit = self.price(ticker) * self.rate(ticker, book.spec.currency)
            share = book.qty[ticker] * unit / invested
            self.trade(book, ticker, -min(book.qty[ticker], float(np.ceil(missing * share / unit))))
        if book.cash < amount:  # rounding left a small gap: sell from the largest line
            largest = max(book.qty, key=lambda t: book.qty[t] * self.price(t))
            unit = self.price(largest) * self.rate(largest, book.spec.currency)
            gap = float(np.ceil((amount - book.cash) / unit)) + 1
            self.trade(book, largest, -min(book.qty[largest], gap))


def _is_quarter_end(dates: pd.DatetimeIndex, i: int) -> bool:
    return i + 1 < len(dates) and dates[i].quarter != dates[i + 1].quarter


def _is_income_date(dates: pd.DatetimeIndex, i: int) -> bool:
    d = dates[i]
    return d.month in (3, 6, 9, 12) and d.day >= 15 and (i == 0 or dates[i - 1].day < 15)


def simulate_transactions(
    prices: pd.DataFrame, fx: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(prices.index)
    sim = _Simulator(prices, fx)
    books = [_Book(spec, _targets(spec, rng)) for spec in CLIENTS]

    flows: dict[str, dict[int, float]] = {}
    for book in books:
        n_flows = int(rng.integers(2, 6))
        days = rng.choice(np.arange(20, len(dates) - 5), size=n_flows, replace=False)
        flows[book.spec.client_id] = {
            int(d): float(rng.choice([-1.0, 1.0], p=[0.4, 0.6]) * rng.uniform(0.03, 0.15))
            for d in days
        }
    initial = {b.spec.client_id: float(rng.integers(20, 250)) * 100_000 for b in books}
    offsets = {b.spec.client_id: int(rng.integers(0, REBALANCE_EVERY)) for b in books}

    for i in range(len(dates)):
        sim.i = i
        for book in books:
            cid = book.spec.client_id
            if i == 0:
                sim.record(book, "deposit", initial[cid])
                sim.rebalance(book)
                continue
            if _is_income_date(dates, i):
                for ticker, q in list(book.qty.items()):
                    y = INSTRUMENTS[ticker].income_yield
                    if y > 0:
                        income = (
                            q * sim.price(ticker) * sim.rate(ticker, book.spec.currency) * y / 4
                        )
                        sim.record(book, "dividend", income, ticker)
            if i in flows[cid]:
                amount = round(flows[cid][i] * sim.value(book), -3)
                if amount > 0:
                    sim.record(book, "deposit", amount)
                    sim.rebalance(book)
                else:
                    sim.raise_cash(book, -amount)
                    sim.record(book, "withdrawal", amount)
            if (i + offsets[cid]) % REBALANCE_EVERY == 0:
                sim.rebalance(book, threshold=0.005)
            if _is_quarter_end(dates, i):
                fee = sim.value(book) * MANAGEMENT_FEE / 4
                sim.raise_cash(book, fee)
                sim.record(book, "fee", -fee)

    rows = [r for b in books for r in b.rows]
    transactions = pd.DataFrame(rows)
    transactions.insert(0, "txn_id", [f"T{n:06d}" for n in range(1, len(rows) + 1)])

    end = dates[-1].date()
    pos_rows: list[dict[str, object]] = []
    for book in books:
        for ticker in sorted(book.qty):
            pos_rows.append(
                {
                    "client_id": book.spec.client_id,
                    "ticker": ticker,
                    "quantity": book.qty[ticker],
                    "as_of": end,
                }
            )
        pos_rows.append(
            {
                "client_id": book.spec.client_id,
                "ticker": "CASH",
                "quantity": book.cash,
                "as_of": end,
            }
        )
    return transactions, pd.DataFrame(pos_rows)


def _long(frame: pd.DataFrame, value: str, key: str) -> pd.DataFrame:
    out = (
        frame.rename_axis("date").reset_index().melt(id_vars="date", var_name=key, value_name=value)
    )
    out["date"] = out["date"].dt.date
    return out.sort_values(["date", key]).reset_index(drop=True)


def generate(seed: int, end: date, *, offline: bool, cache_dir: Path) -> GeneratedData:
    rng = np.random.default_rng(seed)
    dates = business_days(end)
    instruments = list(UNIVERSE)
    prices, synthetic = load_prices(instruments, dates, rng, offline=offline, cache_dir=cache_dir)
    fx = load_fx(dates, rng, offline=offline, cache_dir=cache_dir)
    transactions, positions = simulate_transactions(prices, fx, rng)

    inst_df = pd.DataFrame([i.model_dump(exclude={"mu", "sigma"}) for i in instruments]).assign(
        synthetic_prices=lambda d: d["ticker"].isin(synthetic)
    )
    clients = pd.DataFrame(
        [
            {
                "client_id": c.client_id,
                "name": c.name,
                "banker_id": c.banker_id,
                "currency": c.currency,
                "mandate": c.mandate,
                "profile": c.profile,
            }
            for c in CLIENTS
        ]
    )
    mandates = pd.DataFrame(
        [
            {"mandate": m, "asset_class": ac, "weight": w}
            for m, alloc in MANDATES.items()
            for ac, w in alloc.items()
        ]
    )
    proxies = pd.DataFrame(
        [
            {"asset_class": ac, "ticker": t, "weight": w}
            for ac, sleeve in BENCHMARK_PROXIES.items()
            for t, w in sleeve.items()
        ]
    )
    return GeneratedData(
        instruments=inst_df,
        prices=_long(prices, "close", "ticker"),
        fx=_long(fx, "chf_per_unit", "currency"),
        bankers=pd.DataFrame({"banker_id": list(BANKERS), "name": list(BANKERS.values())}),
        clients=clients,
        mandates=mandates,
        benchmark_proxies=proxies,
        transactions=transactions,
        positions=positions,
        synthetic_tickers=synthetic,
    )
