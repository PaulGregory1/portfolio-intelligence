from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import openai
import pytest

from pi.llm.cache import DiskCache, cache_key
from pi.llm.client import LLMClient, LLMError, strip_reasoning


def raw_response(content: str | None = "hi", tool: tuple[str, str] | None = None) -> Any:
    tool_calls = None
    if tool:
        fn = SimpleNamespace(name=tool[0], arguments=tool[1])
        tool_calls = [SimpleNamespace(id="c1", type="function", function=fn)]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        model="m",
    )


def http_error(
    cls: type[openai.APIStatusError], status: int, retry_after: str | None = None
) -> Exception:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(status, headers=headers, request=httpx.Request("POST", "http://x"))
    return cls("error", response=response, body=None)


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def create(self, **_: Any) -> Any:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(
    outcomes: list[Any], cache: DiskCache | None = None, retries: int = 3
) -> tuple[LLMClient, FakeCompletions]:
    client = LLMClient(
        base_url="http://x",
        api_key="k",
        model="m",
        max_retries=retries,
        cache=cache,
        backoff_base_s=0.0,
        backoff_cap_s=0.0,
    )
    fake = FakeCompletions(outcomes)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))  # type: ignore[assignment]
    return client, fake


def test_parses_content_tool_calls_and_usage() -> None:
    client, _ = make_client([raw_response(None, ("get_portfolio", '{"client_id":"C01"}'))])
    resp = client.complete([{"role": "user", "content": "x"}], tools=[])
    assert resp.tool_calls[0].name == "get_portfolio"
    assert resp.usage.total_tokens == 18
    assert resp.latency_s >= 0


def test_retries_on_rate_limit_and_server_errors() -> None:
    client, fake = make_client(
        [
            http_error(openai.RateLimitError, 429, retry_after="0"),
            http_error(openai.InternalServerError, 503),
            raw_response("ok"),
        ]
    )
    resp = client.complete([{"role": "user", "content": "x"}], tools=None)
    assert resp.content == "ok"
    assert resp.retries == 2
    assert fake.calls == 3


def test_gives_up_after_max_retries() -> None:
    client, fake = make_client([http_error(openai.RateLimitError, 429)] * 3, retries=2)
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "x"}], tools=None)
    assert fake.calls == 3


def test_does_not_retry_client_errors() -> None:
    client, fake = make_client([http_error(openai.BadRequestError, 400)])
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "x"}], tools=None)
    assert fake.calls == 1


def test_disk_cache_avoids_second_call(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    client, fake = make_client([raw_response("first")], cache=cache)
    messages = [{"role": "user", "content": "same question"}]
    first = client.complete(messages, tools=None)
    second = client.complete(messages, tools=None)
    assert fake.calls == 1
    assert not first.cached and second.cached
    assert second.content == "first"


def test_cache_key_depends_on_inputs() -> None:
    base = {"model": "m", "messages": [{"role": "user", "content": "a"}]}
    assert cache_key(base) == cache_key(dict(base))
    assert cache_key(base) != cache_key({**base, "model": "other"})


def test_missing_model_is_an_explicit_error() -> None:
    with pytest.raises(LLMError, match="pi models"):
        LLMClient(base_url="http://x", api_key="k", model="")


def test_strip_reasoning() -> None:
    assert strip_reasoning("<think>hmm\n</think>\nAnswer") == "Answer"
    assert strip_reasoning(None) is None
