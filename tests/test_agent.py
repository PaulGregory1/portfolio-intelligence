from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi.agent.grounding import check_grounding
from pi.agent.loop import Agent, AgentConfig
from pi.data.store import PortfolioData
from pi.llm.client import LLMError, LLMResponse, Message
from pi.tools.access import ToolContext
from pi.tools.portfolio_tools import registry

from .fakes import ScriptedLLM, call, say


def make_agent(
    data: PortfolioData, steps: list[object], banker: str = "B1", **config: object
) -> tuple[Agent, ScriptedLLM]:
    llm = ScriptedLLM(steps)  # type: ignore[arg-type]
    ctx = ToolContext(banker_id=banker, data=data)
    agent = Agent(llm, registry, ctx, config=AgentConfig(**config), grounder=check_grounding)  # type: ignore[arg-type]
    return agent, llm


def test_direct_answer_without_tools(data: PortfolioData) -> None:
    agent, llm = make_agent(data, [say("Bonjour, comment puis-je vous aider ?")])
    result = agent.run("Bonjour")
    assert result.status == "ok"
    assert result.answer.startswith("Bonjour")
    assert result.trace.tool_steps == []
    system = llm.requests[0][0]
    assert system["role"] == "system" and "Anna Keller" in system["content"]
    assert result.trace.grounding is not None and result.trace.grounding["checked"] == 0


def test_single_tool_call_then_grounded_answer(data: PortfolioData) -> None:
    def answer(messages: list[Message]) -> LLMResponse:
        perf = json.loads(messages[-1]["content"])
        return say(f"YTD TWR: {perf['twr_pct']}% vs benchmark {perf['benchmark_return_pct']}%.")

    agent, llm = make_agent(data, [call(("get_performance", {"client_id": "C01"})), answer])
    result = agent.run("Performance YTD de C01 ?")
    assert result.status == "ok"
    assert result.trace.tools_called == ["get_performance"]
    assert result.trace.grounding["grounding_rate"] == 1.0  # type: ignore[index]
    # The tool result is sent back with the matching tool_call_id.
    assert llm.requests[1][-1]["tool_call_id"] == "call_0"
    assert llm.requests[1][-2]["tool_calls"][0]["function"]["name"] == "get_performance"
    assert result.trace.accessed_clients == ["C01"]


def test_several_tool_calls_in_one_response(data: PortfolioData) -> None:
    agent, llm = make_agent(
        data,
        [
            call(
                ("get_performance", {"client_id": "C01"}),
                ("get_risk_report", {"client_id": "C02"}),
                ("list_clients", {}),
            ),
            say("Done."),
        ],
    )
    result = agent.run("Compare C01 and C02")
    assert result.trace.tools_called == ["get_performance", "get_risk_report", "list_clients"]
    assert len(llm.last_tool_results()) == 3
    assert result.trace.iterations == 2


def test_invalid_arguments_are_returned_so_the_model_can_fix_them(data: PortfolioData) -> None:
    agent, llm = make_agent(
        data,
        [
            call(("get_attribution", {"client_id": "C03", "by": "sector"})),
            call(("get_attribution", {"client_id": "C03", "by": "region"})),
            say("Fixed."),
        ],
    )
    result = agent.run("Attribution by sector for C03")
    assert result.status == "ok"
    first, second = result.trace.tool_steps
    assert first.error_code == "INVALID_ARGUMENTS" and not first.ok
    assert second.ok
    assert "by" in llm.requests[1][-1]["content"]


def test_one_json_repair_attempt_then_stop(data: PortfolioData) -> None:
    agent, _ = make_agent(
        data,
        [call(("get_portfolio", "{client_id: C01")), call(("get_portfolio", "{also bad"))],
    )
    result = agent.run("Portfolio of C01")
    assert result.status == "invalid_tool_calls"
    assert [s.error_code for s in result.trace.tool_steps] == ["INVALID_JSON", "INVALID_JSON"]


def test_json_repair_succeeds(data: PortfolioData) -> None:
    agent, _ = make_agent(
        data,
        [
            call(("get_portfolio", "{client_id: C01")),
            call(("get_portfolio", {"client_id": "C01"})),
            say("OK"),
        ],
    )
    assert agent.run("Portfolio of C01").status == "ok"


def test_max_iterations(data: PortfolioData) -> None:
    periods = ["MTD", "QTD", "YTD", "1Y", "3Y", "ITD"]
    steps = [call(("get_performance", {"client_id": "C01", "period": p})) for p in periods]
    agent, _ = make_agent(data, steps, max_iterations=4)  # type: ignore[arg-type]
    result = agent.run("Keep going")
    assert result.status == "max_iterations"
    assert result.trace.iterations == 4
    assert result.answer  # a fallback message, never an empty answer


def test_identical_calls_are_detected(data: PortfolioData) -> None:
    same = call(("get_performance", {"client_id": "C01"}))
    agent, llm = make_agent(data, [same, same, same, same, say("unreachable")])
    result = agent.run("Loop")
    assert result.status == "loop_detected"
    # Executed twice, then refused once with REPEATED_CALL before stopping.
    assert len(result.trace.tool_steps) == 2
    assert llm.last_tool_results()[0]["error"]["code"] == "REPEATED_CALL"


def test_access_denied_is_reported_to_the_model(data: PortfolioData) -> None:
    agent, llm = make_agent(
        data,
        [
            call(("get_portfolio", {"client_id": "C05"})),
            say("Ce client n'est pas dans votre book."),
        ],
    )
    result = agent.run("Portefeuille de Giovanni Bernasconi ?")
    assert result.status == "ok"
    assert result.trace.tool_steps[0].error_code == "ACCESS_DENIED"
    assert llm.last_tool_results()[0]["error"]["code"] == "ACCESS_DENIED"
    assert result.trace.accessed_clients == []


def test_ungrounded_number_triggers_one_correction(data: PortfolioData) -> None:
    def corrected(messages: list[Message]) -> LLMResponse:
        assert "Grounding check failed" in messages[-1]["content"]
        assert "12.34%" in messages[-1]["content"]
        perf = json.loads(messages[-3]["content"])
        return say(f"The YTD return is {perf['twr_pct']}%.")

    agent, _ = make_agent(
        data,
        [
            call(("get_performance", {"client_id": "C01"})),
            say("The YTD return is 12.34%."),
            corrected,
        ],
    )
    result = agent.run("YTD?")
    assert result.status == "ok"
    assert result.trace.grounding_retried
    assert "12.34" not in result.answer


def test_still_ungrounded_after_correction(data: PortfolioData) -> None:
    agent, _ = make_agent(
        data,
        [
            call(("get_performance", {"client_id": "C01"})),
            say("It is 12.34%."),
            say("It is 56.78%."),
        ],
    )
    result = agent.run("YTD?")
    assert result.status == "ungrounded"
    assert result.trace.grounding["ungrounded"] == ["56.78%"]  # type: ignore[index]


def test_grounding_retry_can_be_disabled(data: PortfolioData) -> None:
    agent, _ = make_agent(data, [say("Returns were 9.99%.")], grounding_retry=False)
    assert agent.run("?").status == "ungrounded"


def test_timeout(data: PortfolioData) -> None:
    agent, _ = make_agent(data, [call(("list_clients", {})), say("late")], timeout_s=0.0)
    result = agent.run("slow")
    assert result.status == "timeout"


def test_llm_error_is_caught(data: PortfolioData) -> None:
    def boom(messages: list[Message]) -> LLMResponse:
        raise LLMError("503 Service Unavailable")

    agent, _ = make_agent(data, [boom])
    result = agent.run("?")
    assert result.status == "error"
    assert result.trace.error == "503 Service Unavailable"


def test_trace_is_saved_as_json(data: PortfolioData, tmp_path: Path) -> None:
    agent, _ = make_agent(data, [call(("list_clients", {})), say("You have 3 clients.")])
    result = agent.run("How many clients?")
    path = result.trace.save(tmp_path)
    saved = json.loads(path.read_text())
    assert saved["status"] == "ok"
    assert saved["tools_called"] == ["list_clients"]
    assert saved["prompt_tokens"] == 200
    assert saved["tool_steps"][0]["result"]["banker_id"] == "B1"


def test_unknown_banker_is_rejected(data: PortfolioData) -> None:
    from pi.agent.factory import build_agent
    from pi.config import Settings

    with pytest.raises(ValueError, match="Unknown banker"):
        build_agent(Settings(_env_file=None), data, "B42", llm=ScriptedLLM([]))  # type: ignore[call-arg]


def test_numbers_from_earlier_chat_turns_are_grounded(data: PortfolioData) -> None:
    history: list[Message] = [
        {"role": "user", "content": "Quels sont mes clients ?"},
        {"role": "assistant", "content": "C03 vaut USD 37'085'221.50."},
    ]
    agent, llm = make_agent(data, [say("Pour rappel, C03 vaut USD 37'085'221.50.")])
    result = agent.run("Rappelle-moi la valeur de C03", history=history)
    assert result.status == "ok"
    assert llm.requests[0][1:3] == history
