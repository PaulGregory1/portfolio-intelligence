from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from pi.data.store import PortfolioData
from pi.mcp_server import build_server
from pi.tools.access import ToolContext
from pi.tools.portfolio_tools import registry

EXPECTED_TOOLS = {
    "list_clients",
    "get_portfolio",
    "get_transactions",
    "get_performance",
    "get_attribution",
    "get_risk_report",
    "compare_to_benchmark",
    "get_exposures",
    "screen_universe",
    "search_instruments",
}

# One valid call per tool, on a client of banker B1.
VALID_CALLS: dict[str, dict[str, Any]] = {
    "list_clients": {},
    "get_portfolio": {"client_id": "C01"},
    "get_transactions": {"client_id": "C01", "period": "1Y", "types": ["dividend"]},
    "get_performance": {"client_id": "C02", "period": "2025"},
    "get_attribution": {"client_id": "C03", "period": "2026-Q3", "by": "region"},
    "get_risk_report": {"client_id": "C01"},
    "compare_to_benchmark": {"client_id": "C02", "period": "QTD"},
    "get_exposures": {"client_id": "C03"},
    "screen_universe": {"asset_class": "equity", "min_esg_score": 70},
    "search_instruments": {"query": "gold"},
}

# One invalid call per tool: unknown field, wrong type or out-of-range value.
INVALID_CALLS: dict[str, dict[str, Any]] = {
    "list_clients": {"banker_id": "B2"},
    "get_portfolio": {},
    "get_transactions": {"client_id": "C01", "limit": 0},
    "get_performance": {"client_id": "C01", "start_date": "not a date"},
    "get_attribution": {"client_id": "C01", "by": "sector"},
    "get_risk_report": {"client_id": 12},
    "compare_to_benchmark": {"client_id": "C01", "foo": 1},
    "get_exposures": {"client_id": "C01", "by": "colour"},
    "screen_universe": {"min_esg_score": 140},
    "search_instruments": {"query": ""},
}


@pytest.fixture()
def ctx(data: PortfolioData) -> ToolContext:
    return ToolContext(banker_id="B1", data=data)


def test_registry_exposes_all_tools_in_openai_format() -> None:
    assert set(registry.names()) == EXPECTED_TOOLS
    for spec in registry.openai_tools():
        assert spec["type"] == "function"
        fn = spec["function"]
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert "title" not in json.dumps(params).replace('"title":', "")
    # The banker id is never a parameter the model can set.
    assert "banker_id" not in json.dumps(registry.openai_tools())


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_valid_call_succeeds(ctx: ToolContext, name: str) -> None:
    result = registry.execute(name, VALID_CALLS[name], ctx)
    assert result.ok, result.content
    json.loads(result.to_message_content())


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_invalid_arguments_are_reported_not_raised(ctx: ToolContext, name: str) -> None:
    result = registry.execute(name, INVALID_CALLS[name], ctx)
    assert not result.ok
    assert result.error_code == "INVALID_ARGUMENTS"
    assert result.content["error"]["message"]


def test_access_denied_for_another_bankers_client(ctx: ToolContext) -> None:
    for name in ("get_portfolio", "get_performance", "get_risk_report", "get_transactions"):
        result = registry.execute(name, {"client_id": "C05"}, ctx)
        assert result.error_code == "ACCESS_DENIED"
    # Also when the client is designated by name.
    result = registry.execute("get_portfolio", {"client_id": "Giovanni Bernasconi"}, ctx)
    assert result.error_code == "ACCESS_DENIED"
    assert "C05" not in ctx.accessed_clients


def test_list_clients_only_shows_own_book(data: PortfolioData) -> None:
    for banker_id in data.bankers:
        ctx = ToolContext(banker_id=banker_id, data=data)
        result = registry.execute("list_clients", {}, ctx)
        ids = {c["client_id"] for c in result.content["clients"]}
        assert ids == {c.client_id for c in data.clients_of(banker_id)}


def test_errors_for_unknown_client_tool_and_bad_json(ctx: ToolContext) -> None:
    assert registry.execute("get_portfolio", {"client_id": "C99"}, ctx).error_code == "NOT_FOUND"
    assert registry.execute("get_weather", {}, ctx).error_code == "UNKNOWN_TOOL"
    assert registry.execute("get_portfolio", "{oops", ctx).error_code == "INVALID_JSON"
    assert registry.execute("get_portfolio", "[1]", ctx).error_code == "INVALID_JSON"
    bad_period = registry.execute("get_performance", {"client_id": "C01", "period": "soon"}, ctx)
    assert bad_period.error_code == "INVALID_ARGUMENTS"


def test_client_can_be_referenced_by_name(ctx: ToolContext) -> None:
    result = registry.execute("get_performance", {"client_id": "elena brunner"}, ctx)
    assert result.ok
    assert result.content["client_id"] == "C01"


def test_numbers_are_rounded(ctx: ToolContext) -> None:
    perf = registry.execute("get_performance", {"client_id": "C01", "period": "1Y"}, ctx).content
    for key, value in perf.items():
        if key.endswith("_pct") and value is not None:
            assert round(value, 2) == value
    port = registry.execute("get_portfolio", {"client_id": "C01"}, ctx).content
    assert sum(p["weight_pct"] for p in port["positions"]) == pytest.approx(100.0, abs=0.1)


def test_tool_output_matches_analytics(ctx: ToolContext, data: PortfolioData) -> None:
    from pi.analytics.performance import performance

    out = registry.execute("get_performance", {"client_id": "C02", "period": "YTD"}, ctx).content
    assert out["twr_pct"] == round(100 * performance(data, "C02", "YTD").twr, 2)


def test_mcp_server_lists_and_calls_tools(data: PortfolioData) -> None:
    server = build_server(data, "B1")
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS
    perf = next(t for t in tools if t.name == "get_performance")
    assert "client_id" in perf.input_schema["required"]
    assert "banker_id" not in json.dumps(perf.input_schema)

    result = asyncio.run(server.call_tool("get_performance", {"client_id": "C01"}))
    payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
    assert payload["client_id"] == "C01"
    denied = asyncio.run(server.call_tool("get_performance", {"client_id": "C07"}))
    assert json.loads(denied.content[0].text)["error"]["code"] == "ACCESS_DENIED"  # type: ignore[union-attr]
