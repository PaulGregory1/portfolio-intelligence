from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pi.config import Settings
from pi.data.store import PortfolioData
from pi.web.app import create_app
from tests.fakes import ScriptedLLM, call, say


def make_client(data: PortfolioData, runs: Path, llm: ScriptedLLM | None = None) -> TestClient:
    settings = Settings(runs_dir=runs, llm_model="fake-model")
    return TestClient(create_app(settings, data, llm=llm))


def sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for chunk in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_meta_and_book(data: PortfolioData, tmp_path: Path) -> None:
    client = make_client(data, tmp_path)
    meta = client.get("/api/meta").json()
    assert {b["id"] for b in meta["bankers"]} == set(data.bankers)
    assert meta["models"] == ["fake-model"]

    book = client.get("/api/book/B1").json()
    assert {c["client_id"] for c in book["clients"]} == {c.client_id for c in data.clients_of("B1")}
    first = book["clients"][0]
    assert first["active_ytd_pct"] == pytest.approx(
        first["ytd_pct"] - first["benchmark_ytd_pct"], abs=0.02
    )
    assert sum(first["allocation_pct"].values()) == pytest.approx(100, abs=0.1)
    assert client.get("/api/book/B9").status_code == 404


def test_ask_streams_events_and_saves_the_trace(data: PortfolioData, tmp_path: Path) -> None:
    client_id = data.clients_of("B1")[0].client_id
    llm = ScriptedLLM([call(("get_portfolio", {"client_id": client_id})), say("Done.")])
    client = make_client(data, tmp_path, llm)
    resp = client.post("/api/ask", json={"question": "Positions?", "banker_id": "B1"})
    events = sse(resp.text)
    kinds = [k for k, _ in events]
    assert kinds[:3] == ["llm", "tool_call", "tool_result"]
    kind, trace = events[-1]
    assert kind == "done"
    assert trace["answer"] == "Done." and trace["tools_called"] == ["get_portfolio"]
    assert "messages" not in trace

    replay = client.get(f"/api/traces/{trace['run_id']}").json()
    assert replay["answer"] == "Done."
    assert client.get("/api/traces/nope").status_code == 404


def test_eval_runs_skip_smoke_runs(data: PortfolioData, tmp_path: Path) -> None:
    for name, items in (("full", 102), ("smoke", 3)):
        run = tmp_path / "evals" / name
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({"overall": {"items": items}}))
        (run / "config.json").write_text(json.dumps({"model": f"m-{name}"}))
    client = make_client(data, tmp_path)
    assert [r["run"] for r in client.get("/api/evals").json()] == ["full"]
    assert client.get("/api/meta").json()["models"] == ["fake-model", "m-full"]
