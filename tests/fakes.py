"""Scripted LLM: replays a fixed list of responses, used to test the agent without an API."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pi.llm.client import LLMResponse, Message, ToolCall, Usage

Step = LLMResponse | Callable[[list[Message]], LLMResponse]


def say(text: str) -> LLMResponse:
    return LLMResponse(content=text, usage=Usage(100, 20), latency_s=0.01, model="fake")


def call(*calls: tuple[str, dict[str, Any] | str]) -> LLMResponse:
    """A response asking for one or several tool calls."""
    tool_calls = [
        ToolCall(
            id=f"call_{i}", name=name, arguments=args if isinstance(args, str) else json.dumps(args)
        )
        for i, (name, args) in enumerate(calls)
    ]
    return LLMResponse(content=None, tool_calls=tool_calls, usage=Usage(100, 10), model="fake")


class ScriptedLLM:
    model = "fake-model"

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.requests: list[list[Message]] = []

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> LLMResponse:
        self.requests.append([dict(m) for m in messages])
        if not self.steps:
            raise AssertionError("ScriptedLLM ran out of responses")
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step

    def last_tool_results(self) -> list[dict[str, Any]]:
        """Tool messages the agent sent in the last request."""
        last = self.requests[-1]
        tail = []
        for m in reversed(last):
            if m["role"] != "tool":
                break
            tail.append(json.loads(m["content"]))
        return list(reversed(tail))
