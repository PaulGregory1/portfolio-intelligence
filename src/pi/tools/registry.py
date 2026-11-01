"""Tool registry: pydantic validation, structured errors and export to the OpenAI `tools` format."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from pi.tools.access import ToolContext, ToolFailure
from pi.tools.schemas import Output, ToolInput

ToolFn = Callable[[ToolContext, Any], Output]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[ToolInput]
    output_model: type[Output]
    fn: ToolFn

    def json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = _strip_titles(self.input_model.model_json_schema())
        schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        return schema

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }


def _strip_titles(node: Any) -> Any:
    """Drop the auto-generated `title` keys: they cost tokens and carry no information."""
    if isinstance(node, dict):
        return {
            k: _strip_titles(v)
            for k, v in node.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


@dataclass(frozen=True)
class ToolResult:
    name: str
    arguments: dict[str, Any] | None
    ok: bool
    content: dict[str, Any]
    duration_ms: float

    @property
    def error_code(self) -> str | None:
        return None if self.ok else str(self.content["error"]["code"])

    def to_message_content(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, separators=(",", ":"))


def _error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self, name: str, description: str, input_model: type[ToolInput], output_model: type[Output]
    ) -> Callable[[ToolFn], ToolFn]:
        def decorator(fn: ToolFn) -> ToolFn:
            if name in self._tools:
                raise ValueError(f"Tool {name} registered twice.")
            self._tools[name] = Tool(name, description, input_model, output_model, fn)
            return fn

        return decorator

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Any:
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]

    def execute(
        self, name: str, arguments: str | dict[str, Any] | None, ctx: ToolContext
    ) -> ToolResult:
        """Run a tool. Never raises: every failure becomes a structured error for the model."""
        start = time.perf_counter()

        def done(ok: bool, content: dict[str, Any], args: dict[str, Any] | None) -> ToolResult:
            return ToolResult(name, args, ok, content, (time.perf_counter() - start) * 1000)

        if name not in self._tools:
            return done(
                False,
                _error(
                    "UNKNOWN_TOOL", f"No tool named '{name}'. Available: {', '.join(self._tools)}."
                ),
                None,
            )
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return done(
                    False, _error("INVALID_JSON", f"Arguments are not valid JSON: {exc}"), None
                )
        else:
            parsed = arguments or {}
        if not isinstance(parsed, dict):
            return done(False, _error("INVALID_JSON", "Arguments must be a JSON object."), None)

        tool = self._tools[name]
        try:
            params: BaseModel = tool.input_model.model_validate(parsed)
        except ValidationError as exc:
            return done(False, _error("INVALID_ARGUMENTS", _format_validation(exc)), parsed)
        try:
            output = tool.fn(ctx, params)
        except ToolFailure as exc:
            return done(False, _error(exc.code, exc.message), parsed)
        except ValueError as exc:
            return done(False, _error("INVALID_ARGUMENTS", str(exc)), parsed)
        except Exception as exc:  # a bug in a tool must not crash the agent loop
            return done(False, _error("INTERNAL_ERROR", f"{type(exc).__name__}: {exc}"), parsed)
        return done(True, output.model_dump(mode="json"), parsed)
