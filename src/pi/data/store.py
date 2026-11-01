"""DuckDB persistence and the in-memory view of the book used by the analytics engine."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import duckdb
import pandas as pd

from pi.data.generate import GeneratedData
from pi.data.universe import Instrument

TABLES = (
    "instruments",
    "prices",
    "fx",
    "bankers",
    "clients",
    "mandates",
    "benchmark_proxies",
    "transactions",
    "positions",
)


def write_database(data: GeneratedData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with duckdb.connect(str(path)) as con:
        for name, frame in data.tables().items():
            con.register("frame", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")
            con.unregister("frame")


def read_tables(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `pi data generate` first.")
    with duckdb.connect(str(path), read_only=True) as con:
        return {name: con.execute(f"SELECT * FROM {name}").df() for name in TABLES}


@dataclass(frozen=True)
class Client:
    client_id: str
    name: str
    banker_id: str
    currency: str
    mandate: str
    profile: str


class PortfolioData:
    """Read-only, analysis-friendly view of the database (wide price and FX matrices)."""

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self.tables = tables
        inst = tables["instruments"].drop(columns=["synthetic_prices"])
        inst = inst.astype(object).where(inst.notna(), None)
        self.instruments: dict[str, Instrument] = {
            str(r["ticker"]): Instrument.model_validate({**r, "mu": 0.0, "sigma": 0.0})
            for r in inst.to_dict("records")
        }
        self.clients: dict[str, Client] = {
            str(r["client_id"]): Client(**{str(k): str(v) for k, v in r.items()})
            for r in tables["clients"].to_dict("records")
        }
        self.bankers: dict[str, str] = dict(
            zip(tables["bankers"]["banker_id"], tables["bankers"]["name"], strict=True)
        )
        tx = tables["transactions"].copy()
        tx["date"] = pd.to_datetime(tx["date"])
        self.transactions = tx

    @classmethod
    def load(cls, path: Path) -> PortfolioData:
        return cls(read_tables(path))

    @cached_property
    def prices(self) -> pd.DataFrame:
        frame = self.tables["prices"].pivot(index="date", columns="ticker", values="close")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="date")
        return frame

    @cached_property
    def fx(self) -> pd.DataFrame:
        frame = self.tables["fx"].pivot(index="date", columns="currency", values="chf_per_unit")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="date")
        return frame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.prices.index)

    @property
    def end_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates[-1])

    def mandate_weights(self, mandate: str) -> dict[str, float]:
        m = self.tables["mandates"]
        rows = m[m["mandate"] == mandate]
        return dict(zip(rows["asset_class"], rows["weight"].astype(float), strict=True))

    def benchmark_constituents(self, mandate: str) -> dict[str, float]:
        """Flatten a composite benchmark into instrument weights (summing to 1)."""
        weights = self.mandate_weights(mandate)
        out: dict[str, float] = {}
        for r in self.tables["benchmark_proxies"].to_dict("records"):
            w = weights.get(str(r["asset_class"]), 0.0) * float(r["weight"])
            if w > 0:
                out[str(r["ticker"])] = out.get(str(r["ticker"]), 0.0) + w
        return out

    def prices_in(self, currency: str) -> pd.DataFrame:
        """Price matrix converted into `currency`."""
        rates = pd.DataFrame(
            {t: self.fx[i.currency] / self.fx[currency] for t, i in self.instruments.items()}
        )
        return self.prices * rates[self.prices.columns]

    def client_transactions(self, client_id: str) -> pd.DataFrame:
        return self.transactions[self.transactions["client_id"] == client_id]

    def clients_of(self, banker_id: str) -> list[Client]:
        return [c for c in self.clients.values() if c.banker_id == banker_id]
