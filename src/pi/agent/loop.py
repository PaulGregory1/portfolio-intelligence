"""The agent loop, written by hand: model -> tool calls -> tool results -> model -> ... -> answer.

Guardrails: a maximum number of iterations, a global timeout, detection of identical repeated
tool calls, a single repair attempt for malformed JSON arguments, and an optional grounding
check of the final answer with one correction round.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pi.agent.prompts import GROUNDING_RETRY, REPEATED_CALL, system_prompt
from pi.agent.trace import LLMStep, ToolStep, Trace
from pi.llm.client import ChatModel, LLMError, LLMResponse, Message, ToolCall
from pi.tools.access import ToolContext
from pi.tools.registry import ToolRegistry, ToolResult

Status = Literal[
    "ok", "ungrounded", "max_iterations", "timeout", "loop_detected", "invalid_tool_calls", "error"
]


class GroundingReport(Protocol):
    @property
    def ungrounded_values(self) -> list[str]: ...

    def to_dict(self) -> dict[str, Any]: ...


Grounder = Callable[[str, list[dict[str, Any]]], GroundingReport]
EventHandler = Callable[[str, dict[str, Any]], None]


@dataclass
class AgentConfig:
    max_iterations: int = 8
    timeout_s: float = 180.0
    max_identical_calls: int = 2  # the 3rd identical call stops the run
    max_invalid_json: int = 1  # one repair attempt, then give up
    grounding_retry: bool = True


@dataclass
class AgentResult:
    answer: str
    status: Status
    trace: Trace = field(repr=False)


FALLBACK = {
    "max_iterations": "I could not complete the analysis within the allowed number of steps.",
    "timeout": "The analysis took too long and was stopped.",
    "loop_detected": "The analysis was stopped because the same tool call kept repeating.",
    "invalid_tool_calls": "The model kept producing invalid tool arguments; analysis stopped.",
    "error": "The language model could not be reached.",
}


def _assistant_message(resp: LLMResponse) -> Message:
    message: Message = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in resp.tool_calls
        ]
    return message


def _tool_message(call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _record(trace: Trace, resp: LLMResponse) -> None:
    trace.llm_steps.append(
        LLMStep(
            iteration=len(trace.llm_steps) + 1,
            latency_s=resp.latency_s,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            finish_reason=resp.finish_reason,
            cached=resp.cached,
            retries=resp.retries,
            content=resp.content,
            tool_calls=[tc.__dict__ for tc in resp.tool_calls],
        )
    )


def _call_signature(call: ToolCall) -> str:
    try:
        return f"{call.name}:{json.dumps(json.loads(call.arguments or '{}'), sort_keys=True)}"
    except json.JSONDecodeError:
        return f"{call.name}:{call.arguments}"


class Agent:
    def __init__(
        self,
        llm: ChatModel,
        registry: ToolRegistry,
        ctx: ToolContext,
        config: AgentConfig | None = None,
        grounder: Grounder | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.config = config or AgentConfig()
        self.grounder = grounder
        self.on_event = on_event or (lambda kind, payload: None)

    def _system(self) -> Message:
        data = self.ctx.data
        name = data.bankers.get(self.ctx.banker_id, self.ctx.banker_id)
        return {
            "role": "system",
            "content": system_prompt(self.ctx.banker_id, name, data.end_date.date()),
        }

    def run(self, question: str, history: list[Message] | None = None) -> AgentResult:
        trace = Trace(question=question, banker_id=self.ctx.banker_id, model=self.llm.model)
        trace.history = [str(m["content"]) for m in history or [] if m.get("content")]
        messages: list[Message] = [
            self._system(),
            *(history or []),
            {"role": "user", "content": question},
        ]
        start = time.perf_counter()
        status: Status = "max_iterations"
        answer = ""
        seen: Counter[str] = Counter()
        invalid_json = 0
        tools = self.registry.openai_tools()

        try:
            for iteration in range(1, self.config.max_iterations + 1):
                if time.perf_counter() - start > self.config.timeout_s:
                    status = "timeout"
                    break
                resp = self.llm.complete(messages, tools)
                _record(trace, resp)
                self.on_event("llm", {"iteration": iteration, "latency_s": resp.latency_s})
                messages.append(_assistant_message(resp))

                if not resp.tool_calls:
                    answer = resp.content or ""
                    status = "ok"
                    if self.grounder is not None:
                        answer, status = self._check_grounding(answer, messages, trace, tools)
                    break

                stop: Status | None = None
                for call in resp.tool_calls:
                    signature = _call_signature(call)
                    seen[signature] += 1
                    if seen[signature] > self.config.max_identical_calls + 1:
                        stop = "loop_detected"
                    if seen[signature] > self.config.max_identical_calls:
                        content = json.dumps(
                            {
                                "error": {
                                    "code": "REPEATED_CALL",
                                    "message": REPEATED_CALL.format(name=call.name),
                                }
                            }
                        )
                        messages.append(_tool_message(call.id, content))
                        continue
                    result = self._execute(call, iteration, trace)
                    if result.error_code == "INVALID_JSON":
                        invalid_json += 1
                        if invalid_json > self.config.max_invalid_json:
                            stop = "invalid_tool_calls"
                    messages.append(_tool_message(call.id, result.to_message_content()))
                if stop:
                    status = stop
                    break
        except LLMError as exc:
            status = "error"
            trace.error = str(exc)

        if status in FALLBACK:
            answer = answer or FALLBACK[status]
        trace.answer = answer
        trace.status = status
        trace.messages = messages
        trace.accessed_clients = list(dict.fromkeys(self.ctx.accessed_clients))
        trace.wall_time_s = time.perf_counter() - start
        self.on_event("final", {"status": status})
        return AgentResult(answer=answer, status=status, trace=trace)

    def _execute(self, call: ToolCall, iteration: int, trace: Trace) -> ToolResult:
        self.on_event("tool_call", {"name": call.name, "arguments": call.arguments})
        result = self.registry.execute(call.name, call.arguments, self.ctx)
        trace.tool_steps.append(
            ToolStep(
                iteration=iteration,
                call_id=call.id,
                name=call.name,
                raw_arguments=call.arguments,
                arguments=result.arguments,
                ok=result.ok,
                error_code=result.error_code,
                result=result.content,
                duration_ms=result.duration_ms,
            )
        )
        self.on_event(
            "tool_result",
            {
                "name": call.name,
                "ok": result.ok,
                "error_code": result.error_code,
                "duration_ms": result.duration_ms,
            },
        )
        return result

    def _check_grounding(
        self, answer: str, messages: list[Message], trace: Trace, tools: list[dict[str, Any]]
    ) -> tuple[str, Status]:
        """Verify the numbers of the answer; ask the model once to fix ungrounded ones."""
        assert self.grounder is not None
        report = self.grounder(answer, trace.grounding_sources())
        trace.grounding = report.to_dict()
        if not report.ungrounded_values:
            return answer, "ok"
        if not self.config.grounding_retry:
            return answer, "ungrounded"

        trace.grounding_retried = True
        self.on_event("grounding_retry", {"numbers": report.ungrounded_values})
        numbers = ", ".join(report.ungrounded_values)
        messages.append({"role": "user", "content": GROUNDING_RETRY.format(numbers=numbers)})
        resp = self.llm.complete(messages, tools)
        _record(trace, resp)
        messages.append(_assistant_message(resp))
        if resp.tool_calls or not resp.content:
            # The correction round does not allow more tool calls: keep the first answer.
            return answer, "ungrounded"
        report = self.grounder(resp.content, trace.grounding_sources())
        trace.grounding = report.to_dict()
        return resp.content, "ok" if not report.ungrounded_values else "ungrounded"
