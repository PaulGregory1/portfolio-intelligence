from __future__ import annotations

import pytest

from pi.data.store import PortfolioData
from pi.data.universe import INSTRUMENTS, UNIVERSE
from pi.screening.filters import ScreenCriteria, instrument_stats, screen
from pi.screening.search import BM25Searcher, expand_query, normalise


@pytest.fixture(scope="module")
def searcher() -> BM25Searcher:
    return BM25Searcher(list(UNIVERSE))


def top(searcher: BM25Searcher, query: str, k: int = 3) -> list[str]:
    return [t for t, _ in searcher.search(query, k)]


def test_normalise_strips_accents_stopwords_and_plurals() -> None:
    assert normalise("Les obligations suisses peu chères") == ["obligation", "suisse", "chere"]
    assert "bond" in expand_query("obligations")
    assert "gold" in expand_query("de l'or")


def test_cheap_swiss_bonds_in_french(searcher: BM25Searcher) -> None:
    hits = top(searcher, "obligations suisses peu chères")
    for ticker in hits:
        inst = INSTRUMENTS[ticker]
        assert inst.asset_class == "fixed_income"
        assert inst.currency == "CHF"


@pytest.mark.parametrize(
    ("query", "check"),
    [
        ("gold", lambda i: i.sub_asset_class == "gold"),
        ("de l'or physique", lambda i: i.sub_asset_class == "gold"),
        ("actions durables ESG", lambda i: i.esg_score >= 80),
        ("emerging markets debt", lambda i: i.region == "Emerging Markets"),
        ("immobilier suisse", lambda i: i.sub_asset_class == "real_estate"),
        ("US inflation protection", lambda i: i.sub_asset_class == "inflation_linked"),
        ("semiconductor", lambda i: i.ticker == "ASML"),
    ],
)
def test_known_queries(searcher: BM25Searcher, query: str, check: object) -> None:
    best = top(searcher, query, 1)
    assert best, query
    assert check(INSTRUMENTS[best[0]]), (query, best)  # type: ignore[operator]


def test_irrelevant_query_returns_nothing(searcher: BM25Searcher) -> None:
    assert searcher.search("zzzz qwerty", 5) == []


def test_screen_filters_combine(data: PortfolioData) -> None:
    stats = instrument_stats(data)
    assert len(stats) == len(UNIVERSE)
    result = screen(
        stats, ScreenCriteria(asset_class="fixed_income", currency="chf", max_ter_pct=0.15)
    )
    assert set(result["ticker"]) == {"CSBGC0", "CSBGC7", "CHCORP"}
    esg = screen(stats, ScreenCriteria(min_esg_score=80, sort_by="esg_score"))
    assert esg["esg_score"].is_monotonic_decreasing
    assert (esg["esg_score"] >= 80).all()
    low_vol = screen(stats, ScreenCriteria(asset_class="equity", max_volatility_pct=15))
    assert (low_vol["volatility_1y_pct"] <= 15).all()
    assert screen(stats, ScreenCriteria(region="Mars")).empty
