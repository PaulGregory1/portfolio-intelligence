"""Run traces: everything needed to replay, debug or score an agent run, saved as JSON."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class LLMStep:
    iteration: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    cached: bool
    retries: int
    content: str | None
    tool_calls: list[dict[str, str]]


@dataclass
class ToolStep:
    iteration: int
    call_id: str
    name: str
    raw_arguments: str
    arguments: dict[str, Any] | None
    ok: bool
    error_code: str | None
    result: dict[str, Any]
    duration_ms: float


@dataclass
class Trace:
    question: str
    banker_id: str
    model: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    llm_steps: list[LLMStep] = field(default_factory=list)
    tool_steps: list[ToolStep] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    status: str = "running"
    grounding: dict[str, Any] | None = None
    grounding_retried: bool = False
    # Earlier turns of a chat session (already checked when they were answered).
    history: list[str] = field(default_factory=list)
    accessed_clients: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0
    error: str | None = None

    @property
    def iterations(self) -> int:
        return len(self.llm_steps)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.llm_steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.llm_steps)

    @property
    def tools_called(self) -> list[str]:
        return [s.name for s in self.tool_steps]

    def tool_outputs(self) -> list[dict[str, Any]]:
        """Successful tool results: the only legitimate source of numbers."""
        return [s.result for s in self.tool_steps if s.ok]

    def grounding_sources(self) -> list[dict[str, Any]]:
        """Tool results, the question and earlier chat turns: numbers that may be repeated."""
        return [{"question": self.question, "history": self.history}, *self.tool_outputs()]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            iterations=self.iterations,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            tools_called=self.tools_called,
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        fields["llm_steps"] = [LLMStep(**s) for s in data.get("llm_steps", [])]
        fields["tool_steps"] = [ToolStep(**s) for s in data.get("tool_steps", [])]
        return cls(**fields)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at[:19].replace(":", "").replace("-", "")
        path = directory / f"{stamp}_{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str))
        return path
