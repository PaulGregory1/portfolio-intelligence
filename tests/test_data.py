from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pi.data.generate import CLIENTS, MANDATES, GeneratedData, generate
from pi.data.prices import business_days
from pi.data.store import PortfolioData, read_tables
from pi.data.universe import UNIVERSE

from .conftest import END, SEED


def test_universe_metadata_is_complete() -> None:
    assert 35 <= len(UNIVERSE) <= 50
    assert len({i.ticker for i in UNIVERSE}) == len(UNIVERSE)
    for inst in UNIVERSE:
        assert 0 <= inst.esg_score <= 100
        assert inst.description.count(".") >= 2


def test_same_seed_gives_identical_database(generated: GeneratedData, tmp_path: Path) -> None:
    again = generate(SEED, END, offline=True, cache_dir=tmp_path)
    for name, frame in generated.tables().items():
        pd.testing.assert_frame_equal(frame, again.tables()[name], obj=name)


def test_different_seed_gives_different_data(generated: GeneratedData, tmp_path: Path) -> None:
    other = generate(SEED + 1, END, offline=True, cache_dir=tmp_path)
    assert not generated.prices["close"].equals(other.prices["close"])


def test_three_years_of_business_days(data: PortfolioData) -> None:
    assert len(data.dates) == len(business_days(END))
    assert data.end_date == pd.Timestamp(END)
    assert data.prices.notna().all().all()
    assert (data.fx["CHF"] == 1.0).all()


def test_positions_are_the_replay_of_transactions(data: PortfolioData) -> None:
    tx = data.transactions
    signed = tx["quantity"].where(tx["type"] == "buy", -tx["quantity"])
    trades = tx[tx["type"].isin(["buy", "sell"])].assign(signed=signed)
    replayed = trades.groupby(["client_id", "ticker"])["signed"].sum()
    replayed = replayed[replayed != 0]
    cash = tx.groupby("client_id")["amount"].sum()

    stored = data.tables["positions"]
    held = stored[stored["ticker"] != "CASH"].set_index(["client_id", "ticker"])["quantity"]
    pd.testing.assert_series_equal(
        replayed.sort_index(), held.sort_index(), check_names=False, check_index_type=False
    )
    stored_cash = stored[stored["ticker"] == "CASH"].set_index("client_id")["quantity"]
    pd.testing.assert_series_equal(
        cash.sort_index(), stored_cash.sort_index(), check_names=False, atol=1e-4
    )


def test_cash_never_goes_negative(data: PortfolioData) -> None:
    for _, g in data.transactions.groupby("client_id"):
        assert g.sort_values(["date", "txn_id"])["amount"].cumsum().min() >= 0


def test_book_structure(data: PortfolioData) -> None:
    assert len(data.bankers) == 3
    assert len(data.clients) == 8
    for banker_id in data.bankers:
        assert 2 <= len(data.clients_of(banker_id)) <= 4
    positions = data.tables["positions"]
    for client_id in data.clients:
        lines = positions[(positions["client_id"] == client_id) & (positions["ticker"] != "CASH")]
        assert 8 <= len(lines) <= 20, client_id
    kinds = set(data.transactions["type"])
    assert kinds == {"deposit", "withdrawal", "buy", "sell", "dividend", "fee"}


@pytest.mark.parametrize("spec", CLIENTS, ids=lambda s: s.client_id)
def test_client_allocation_deviates_from_mandate(spec: object) -> None:
    from pi.data.generate import ClientSpec

    assert isinstance(spec, ClientSpec)
    assert abs(sum(spec.allocation.values()) - 1) < 1e-9
    mandate = MANDATES[spec.mandate]
    assert any(abs(spec.allocation[k] - mandate[k]) >= 0.02 for k in spec.allocation)


def test_benchmark_constituents_sum_to_one(data: PortfolioData) -> None:
    for mandate in MANDATES:
        assert sum(data.benchmark_constituents(mandate).values()) == pytest.approx(1.0)


def test_database_round_trip(db_path: Path, generated: GeneratedData) -> None:
    tables = read_tables(db_path)
    assert len(tables["transactions"]) == len(generated.transactions)
