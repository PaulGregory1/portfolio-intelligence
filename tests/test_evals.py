from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from pi.agent.trace import ToolStep, Trace
from pi.analytics.performance import performance
from pi.data.store import PortfolioData
from pi.evals.dataset import EvalItem, ExpectedValue, build_dataset, load_dataset, save_dataset
from pi.evals.metrics import aggregate, is_refusal, score_item, tool_scores, value_found
from pi.evals.report import compare_runs
from pi.evals.runner import RunConfig, run_eval
from pi.llm.client import LLMResponse, Message

from .fakes import call, say


@pytest.fixture(scope="module")
def items(data: PortfolioData) -> list[EvalItem]:
    return build_dataset(data, seed=7)


def test_dataset_shape(items: list[EvalItem], data: PortfolioData) -> None:
    assert 80 <= len(items) <= 120
    counts = Counter(i.category for i in items)
    assert set(counts) == {
        "lookup",
        "performance",
        "attribution",
        "risk",
        "benchmark",
        "screening",
        "multi_step",
        "access_control",
        "out_of_scope",
    }
    assert len({i.id for i in items}) == len(items)
    assert {i.language for i in items} == {"fr", "en"}
    for item in items:
        assert item.banker_id in data.bankers
        if item.category == "access_control":
            assert item.expected_behavior == "refuse"
            assert data.clients[item.client_id].banker_id != item.banker_id  # type: ignore[index]
        elif item.expected_behavior == "answer":
            assert item.expected_values or item.expected_mentions, item.id


def test_dataset_is_deterministic_and_round_trips(
    items: list[EvalItem], data: PortfolioData, tmp_path: Path
) -> None:
    assert [i.to_json() for i in build_dataset(data, seed=7)] == [i.to_json() for i in items]
    path = tmp_path / "ds.jsonl"
    save_dataset(items, path)
    assert [i.to_json() for i in load_dataset(path)] == [i.to_json() for i in items]


def test_ground_truth_comes_from_the_engine(items: list[EvalItem], data: PortfolioData) -> None:
    perf_items = [
        i
        for i in items
        if (i.category == "performance" and "YTD" in i.id)
        or (i.category == "multi_step" and i.question.startswith("Compare"))
    ]
    item = next(i for i in items if i.category == "multi_step" and "YTD" in i.question)
    names = {c.name: c.client_id for c in data.clients.values()}
    in_question = [cid for name, cid in names.items() if name in item.question]
    expected = sorted(v.value for v in item.expected_values)
    computed = sorted(round(100 * performance(data, cid, "YTD").twr, 2) for cid in in_question)
    assert expected == computed
    assert perf_items


def test_value_found_and_tool_scores() -> None:
    pct = ExpectedValue("twr", 2.95, 0.06, "percent")
    assert value_found(pct, "Le TWR est de 2,95 % YTD.")
    assert value_found(pct, "about 3.0%")
    assert not value_found(pct, "about 3.1%")
    amount = ExpectedValue("v", 4494997.78, 8990.0, "amount")
    assert value_found(amount, "CHF 4.49 Mio")
    assert not value_found(amount, "CHF 4.6 Mio")

    assert tool_scores(["get_performance"], ["list_clients", "get_performance"]) == (1.0, 1.0)
    assert tool_scores(["get_performance", "get_performance"], ["get_performance"]) == (1.0, 0.5)
    assert tool_scores(["get_portfolio|list_clients"], ["list_clients"]) == (1.0, 1.0)
    assert tool_scores(["get_attribution"], ["get_risk_report"]) == (0.0, 0.0)
    assert tool_scores([], ["get_portfolio"]) == (None, None)


def test_refusal_detection() -> None:
    assert is_refusal("Désolé, ce client n'est pas dans votre portefeuille.")
    assert is_refusal("I'm sorry, but I can't provide investment advice.")
    assert is_refusal("Je ne peux pas faire de prévision de marché.")
    assert is_refusal("Giovanni Bernasconi ne figure pas parmi les clients de Luca Rossi.")
    assert is_refusal("« Hoffmann Family Office » n’apparaît pas dans la liste des clients.")
    assert is_refusal("Clara Meier is not one of your clients.")
    assert not is_refusal("The YTD return is 2.95%.")


def _trace(answer: str, status: str = "ok", accessed: list[str] | None = None) -> Trace:
    trace = Trace(question="q", banker_id="B1", model="m", answer=answer, status=status)
    trace.accessed_clients = accessed or []
    trace.grounding = {"grounding_rate": 1.0, "ungrounded": []}
    trace.tool_steps.append(ToolStep(1, "c", "get_performance", "{}", {}, True, None, {}, 1.0))
    return trace


def test_score_item_answer_and_refusal() -> None:
    owners = {"C01": "B1", "C05": "B2"}
    item = EvalItem(
        "performance-01",
        "?",
        "B1",
        "performance",
        "en",
        ["get_performance"],
        [ExpectedValue("twr", 2.95, 0.06, "percent")],
        client_id="C01",
    )
    good = score_item(item, _trace("TWR 2.95%", accessed=["C01"]), owners)
    assert good.success and good.numeric_accuracy == 1.0 and good.access_violations == 0
    bad = score_item(item, _trace("TWR 4.10%"), owners)
    assert not bad.success

    refuse = EvalItem(
        "access_control-01",
        "?",
        "B1",
        "access_control",
        "en",
        [],
        expected_behavior="refuse",
        client_id="C05",
    )
    assert score_item(refuse, _trace("Sorry, C05 is not in your book."), owners).success
    leaked = score_item(refuse, _trace("Sorry...", accessed=["C05"]), owners)
    assert not leaked.success and leaked.access_violations == 1

    agg = aggregate([good.to_dict(), bad.to_dict()])
    assert agg["success_rate"] == 0.5
    assert agg["latency_p50_s"] == 0.0


class CountingLLM:
    """Answers every question by listing clients then politely declining."""

    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], tools: object) -> LLMResponse:
        self.calls += 1
        if messages[-1]["role"] == "user":
            return call(("list_clients", {}))
        return say("Sorry, I cannot help with that request.")


def test_run_eval_end_to_end_and_resume(
    items: list[EvalItem], data: PortfolioData, tmp_path: Path
) -> None:
    by_cat = {
        c: [i for i in items if i.category == c]
        for c in ("lookup", "access_control", "out_of_scope")
    }
    subset = [i for k in range(4) for c in by_cat for i in by_cat[c][k : k + 1]]
    out = tmp_path / "run"
    llm = CountingLLM()
    config = RunConfig(
        model="fake-model", base_url="http://fake", dataset=Path("x"), out_dir=out, concurrency=3
    )
    first = run_eval(subset[:8], data, config, lambda _: llm)
    assert first["overall"]["items"] == 8
    calls_after_first = llm.calls

    summary = run_eval(subset, data, config, lambda _: llm)
    # Only the 4 new items were run (2 model calls each).
    assert llm.calls - calls_after_first == 2 * (len(subset) - 8)
    assert summary["overall"]["items"] == len(subset)
    assert summary["overall"]["access_violations"] == 0
    refusals = summary["by_category"]["access_control"]
    assert refusals["success_rate"] == 1.0
    assert summary["by_category"]["lookup"]["success_rate"] == 0.0

    for name in ("results.jsonl", "summary.json", "report.md", "config.json"):
        assert (out / name).exists()
    assert len(list((out / "traces").glob("*.json"))) == len(subset)
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == len(subset)
    assert "# Evaluation report" in (out / "report.md").read_text()

    table = compare_runs([out, out])
    assert "| Success rate |" in table and "## Success rate by category" in table


def test_mentions_ignore_separator_variants() -> None:
    from pi.evals.metrics import mention_found

    alternatives = ["fixed income", "bond"]
    assert mention_found(alternatives, "The **fixed‑income** sleeve")  # U+2011
    assert mention_found(alternatives, "| fixed_income | -0.11 % |")
    assert mention_found(alternatives, "Fixed-Income")
    assert not mention_found(alternatives, "equity")


def test_rescore_reproduces_scores_from_traces(
    items: list[EvalItem], data: PortfolioData, tmp_path: Path
) -> None:
    from pi.evals.runner import rescore

    subset = [i for i in items if i.category in ("access_control", "lookup")][:4]
    config = RunConfig(
        model="fake-model",
        base_url="http://fake",
        dataset=Path("x"),
        out_dir=tmp_path,
        concurrency=1,
    )
    before = run_eval(subset, data, config, lambda _: CountingLLM())
    after = rescore(subset, data, tmp_path)
    assert after["overall"] == before["overall"]
