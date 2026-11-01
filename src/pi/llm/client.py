"""Chat-completion client for any OpenAI-compatible endpoint.

Adds what the agent and the evaluation need on top of the raw SDK: retries with exponential
backoff and jitter (honouring Retry-After), a process-wide concurrency limit, an optional disk
cache, token accounting and latency measurement.
"""

from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Protocol

import openai
from openai import OpenAI

from pi.config import Settings
from pi.llm.cache import DiskCache, cache_key

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, exactly as produced by the model


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    model: str = ""
    finish_reason: str | None = None
    cached: bool = False
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMResponse:
        return cls(
            content=data["content"],
            tool_calls=[ToolCall(**tc) for tc in data["tool_calls"]],
            usage=Usage(**data["usage"]),
            latency_s=data["latency_s"],
            model=data["model"],
            finish_reason=data.get("finish_reason"),
        )


class ChatModel(Protocol):
    """What the agent loop needs from a model. Implemented by `LLMClient` and test fakes."""

    model: str

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str | None) -> str | None:
    """Remove `<think>...</think>` blocks that some reasoning models put in the content."""
    if text is None:
        return None
    return _THINK.sub("", text).strip()


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, openai.APITimeoutError | openai.APIConnectionError | openai.RateLimitError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


class LLMClient:
    # One semaphore per process, shared by every client: the rate limit is per API key.
    _semaphores: ClassVar[dict[int, threading.BoundedSemaphore]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout_s: float = 60.0,
        max_retries: int = 5,
        max_concurrency: int = 2,
        cache: DiskCache | None = None,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
    ) -> None:
        if not model:
            raise LLMError("No model configured: set LLM_MODEL (see `pi models`) or pass --model.")
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cache = cache
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        # Retries are handled here (with logging and Retry-After), not inside the SDK.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        with self._lock:
            self._semaphore = self._semaphores.setdefault(
                max_concurrency, threading.BoundedSemaphore(max_concurrency)
            )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        model: str | None = None,
        base_url: str | None = None,
        use_cache: bool = False,
    ) -> LLMClient:
        return cls(
            base_url=base_url or settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=model or settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            max_concurrency=settings.llm_max_concurrency,
            cache=DiskCache(settings.cache_dir / "llm") if use_cache else None,
        )

    def _payload(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> LLMResponse:
        payload = self._payload(messages, tools)
        key = cache_key({**payload, "base_url": self.base_url})
        if self.cache and (hit := self.cache.get(key)):
            cached = LLMResponse.from_dict(hit)
            return LLMResponse(**{**cached.__dict__, "cached": True})

        attempt = 0
        while True:
            start = time.perf_counter()
            try:
                with self._semaphore:
                    raw = self._client.chat.completions.create(**payload)
                break
            except Exception as exc:
                if not _is_retryable(exc) or attempt >= self.max_retries:
                    raise LLMError(f"{type(exc).__name__}: {exc}") from exc
                delay = _retry_after(exc)
                if delay is None:
                    delay = min(self.backoff_cap_s, self.backoff_base_s * 2**attempt)
                    delay *= random.uniform(0.5, 1.5)  # jitter avoids synchronised retries
                time.sleep(delay)
                attempt += 1
        latency = time.perf_counter() - start

        choice = raw.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "")
            for tc in (msg.tool_calls or [])
            if tc.type == "function"
        ]
        usage = Usage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
        )
        response = LLMResponse(
            content=strip_reasoning(msg.content),
            tool_calls=tool_calls,
            usage=usage,
            latency_s=latency,
            model=raw.model or self.model,
            finish_reason=choice.finish_reason,
            retries=attempt,
        )
        if self.cache:
            self.cache.set(key, response.to_dict())
        return response
