from __future__ import annotations

import pytest

from pi.agent.grounding import check_grounding, collect_values, extract_numbers


def values(text: str) -> list[float]:
    return [n.value for n in extract_numbers(text) if n.ignored is None]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CHF 1'234'567.89", 1234567.89),
        ("CHF 1’234’567.89", 1234567.89),
        ("USD 1,234,567.89", 1234567.89),
        ("EUR 1.234.567,89", 1234567.89),
        ("1 234 567,89 €", 1234567.89),
        ("1 234 567,89 EUR", 1234567.89),
        ("CHF 4\u202f494\u202f997.78", 4494997.78),
        ("\u20111,29 %", -1.29),
        ("un rendement de 4,23 %", 4.23),
        ("a return of 4.23%", 4.23),
        ("down -3.25%", -3.25),
        ("en baisse de −3,25 %", -3.25),
        ("up +0.58%", 0.58),
        ("4.49 Mio. CHF", 4_490_000),
        ("4.5M USD", 4_500_000),
        ("CHF 33.8 millions", 33_800_000),
        ("1.2 Mrd", 1_200_000_000),
        ("2.5 bn", 2_500_000_000),
        ("12.5k", 12_500),
        ("+3,735 M EUR", 3_735_000),
        ("549 bps", 5.49),
        ("25 pb", 0.25),
        ("0.135", 0.135),
        ("4 percent", 4.0),
    ],
)
def test_number_formats(text: str, expected: float) -> None:
    assert values(text) == [pytest.approx(expected)]


@pytest.mark.parametrize(
    "text",
    [
        "au 2026-09-30",
        "le 30.09.2026",
        "le 30/09/2026",
        "en 2025",
        "au T3 2026",
        "Q3",
        "sur 1Y",
        "over 3M",
        "les 8 positions",
        "top 5",
        "client C03",
        "banker B1",
        "VaR 95% à 1 jour",
        "99% CVaR",
        "2026-09",
        "(30 Jun 2026 → 30 Sep 2026, EUR)",
        "as of 2026\u201110\u201130",
        "(Thomas Weber – EUR, au 30 oct 2026)",
        "The 1\u2011day 95 % historical VaR",
    ],
)
def test_non_financial_numbers_are_ignored(text: str) -> None:
    assert values(text) == []


def test_mixed_sentence() -> None:
    text = (
        "Au T3 2026 (du 2026-06-30 au 2026-09-30), le portefeuille de C03 a perdu 1,29 % "
        "contre +1,86 % pour le benchmark, soit 3,14 points de sous-performance sur "
        "USD 37'085'221.40."
    )
    assert values(text) == [pytest.approx(v) for v in (1.29, 1.86, 3.14, 37085221.40)]


def test_collect_values_walks_nested_outputs() -> None:
    payload = [{"a": 1, "b": {"c": [2.5, {"d": -3}]}, "flag": True, "s": "S&P 500 at 4.5"}]
    assert collect_values(payload) == [1.0, 2.5, -3.0, 500.0, 4.5]


def test_numbers_in_tool_text_fields_are_grounded() -> None:
    tools = [{"name": "SPDR S&P 500 ETF Trust", "weight_pct": 15.9}]
    assert check_grounding("SPY (S&P 500) weighs 15.9%.", tools).grounding_rate == 1.0


TOOL_OUTPUT = [
    {
        "twr_pct": 2.95,
        "benchmark_return_pct": -2.54,
        "active_return_pct": 5.49,
        "end_value": 4494997.78,
        "weights": {"equity": 27.4},
        "fraction": 0.0423,
    }
]


def test_grounded_answer() -> None:
    answer = (
        "Le portefeuille a gagné 2.95% YTD contre -2,54 % pour le benchmark (surperformance de "
        "5,49 points, soit 549 bps). Valeur: CHF 4'494'997.78 (environ 4.5 Mio.), actions 27%."
    )
    report = check_grounding(answer, TOOL_OUTPUT)
    assert report.ungrounded_values == []
    assert report.grounding_rate == 1.0


def test_sign_carried_by_words_is_accepted() -> None:
    report = check_grounding("underperformed its benchmark by 2.54%", TOOL_OUTPUT)
    assert report.grounding_rate == 1.0


def test_percent_matches_decimal_tool_value() -> None:
    assert check_grounding("a weight of 4.23%", TOOL_OUTPUT).grounding_rate == 1.0


def test_invented_numbers_are_flagged() -> None:
    answer = "The return was 3.10% and the value CHF 4'600'000; volatility is 7.85%."
    report = check_grounding(answer, TOOL_OUTPUT)
    assert report.ungrounded_values == ["3.10%", "4'600'000", "7.85%"]
    assert report.grounding_rate == 0.0


def test_rounding_tolerance_follows_display_precision() -> None:
    # 2.95 rounds to 3.0 (1 decimal) and 3 (0 decimals) but is not 3.1 or 2.9x.
    assert check_grounding("3.0%", TOOL_OUTPUT).grounding_rate == 1.0
    assert check_grounding("3%", TOOL_OUTPUT).grounding_rate == 1.0
    assert check_grounding("3.1%", TOOL_OUTPUT).grounding_rate == 0.0
    assert check_grounding("2.97%", TOOL_OUTPUT).grounding_rate == 0.0
    # 4.49 M is within the precision of 4494997.78, 4.40 M is not.
    assert check_grounding("CHF 4.49M", TOOL_OUTPUT).grounding_rate == 1.0
    assert check_grounding("CHF 4.40M", TOOL_OUTPUT).grounding_rate == 0.0


def test_answer_without_numbers_has_no_rate() -> None:
    report = check_grounding("Ce client n'est pas dans votre portefeuille.", TOOL_OUTPUT)
    assert report.grounding_rate is None
    assert report.to_dict()["checked"] == 0


def test_numbers_from_the_question_are_grounded(data: object) -> None:
    from pi.agent.trace import Trace

    trace = Trace(question="Where will the S&P 500 be next year?", banker_id="B1", model="m")
    report = check_grounding("I cannot forecast the S&P 500.", trace.grounding_sources())
    assert report.ungrounded_values == []
