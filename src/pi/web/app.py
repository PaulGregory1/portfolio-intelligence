"""HTTP API behind `pi ui`. The agent runs in a worker thread and streams its events (SSE).

Endpoints:
    GET  /api/meta              bankers, default model, models seen in evaluation runs
    GET  /api/book/{banker}     clients of a banker with YTD performance and allocation
    POST /api/ask               run the agent, stream tool calls, then the final trace
    GET  /api/traces/{run_id}   a saved trace (chat or evaluation), for replay
    GET  /api/evals             summaries of the evaluation runs under runs/evals
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pi.agent.factory import build_agent
from pi.agent.grounding import check_grounding
from pi.agent.trace import Trace
from pi.config import Settings
from pi.data.store import PortfolioData
from pi.llm.client import ChatModel
from pi.tools.access import ToolContext
from pi.tools.portfolio_tools import registry

STATIC = Path(__file__).parent / "static"


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    banker_id: str
    model: str | None = None
    history: list[Turn] = Field(default_factory=list)


def trace_payload(trace: Trace) -> dict[str, Any]:
    """Everything the interface shows; the raw message list stays in the saved file."""
    data = trace.to_dict()
    data.pop("messages", None)
    return data


def _tool(name: str, ctx: ToolContext, **arguments: Any) -> dict[str, Any]:
    result = registry.execute(name, json.dumps(arguments), ctx)
    if not result.ok:
        raise HTTPException(500, f"{name} failed: {result.content['error']['message']}")
    return result.content


def book(data: PortfolioData, banker_id: str) -> dict[str, Any]:
    """The banker's clients, computed through the same tools the agent uses."""
    ctx = ToolContext(banker_id=banker_id, data=data)
    listing = _tool("list_clients", ctx)
    clients = []
    for c in listing["clients"]:
        perf = _tool("get_performance", ctx, client_id=c["client_id"], period="YTD")
        expo = _tool("get_exposures", ctx, client_id=c["client_id"], by="asset_class")
        clients.append(
            {
                **c,
                "ytd_pct": perf["twr_pct"],
                "benchmark_ytd_pct": perf["benchmark_return_pct"],
                "active_ytd_pct": perf["active_return_pct"],
                "allocation_pct": expo["exposures_pct"]["asset_class"],
            }
        )
    return {**listing, "clients": clients}


def eval_runs(runs_dir: Path) -> list[dict[str, Any]]:
    out = []
    for summary in sorted((runs_dir / "evals").glob("*/summary.json")):
        config_path = summary.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        data = json.loads(summary.read_text())
        if data.get("overall", {}).get("items", 0) < 10:
            continue  # smoke runs
        out.append({"run": summary.parent.name, "model": config.get("model", "?"), **data})
    return out


def find_trace(runs_dir: Path, run_id: str) -> Path | None:
    if not run_id.isalnum():
        return None
    for pattern in (f"traces/*_{run_id}.json", f"evals/*/traces/*_{run_id}.json"):
        if hits := sorted(runs_dir.glob(pattern)):
            return hits[0]
    return None


def create_app(
    settings: Settings,
    data: PortfolioData,
    *,
    use_cache: bool = False,
    llm: ChatModel | None = None,
) -> FastAPI:
    """`llm` replaces the configured endpoint (tests use a scripted model)."""
    app = FastAPI(title="Portfolio Intelligence", docs_url="/api/docs")
    book_cache: dict[str, dict[str, Any]] = {}

    def check_banker(banker_id: str) -> None:
        if banker_id not in data.bankers:
            raise HTTPException(404, f"Unknown banker '{banker_id}'.")

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        models = [settings.llm_model, *(r["model"] for r in eval_runs(settings.runs_dir))]
        return {
            "bankers": [{"id": k, "name": v} for k, v in data.bankers.items()],
            "model": settings.llm_model,
            "models": [m for m in dict.fromkeys(models) if m],
            "as_of_date": data.end_date.date().isoformat(),
            "base_url": settings.llm_base_url,
        }

    @app.get("/api/book/{banker_id}")
    def get_book(banker_id: str) -> dict[str, Any]:
        check_banker(banker_id)
        if banker_id not in book_cache:
            book_cache[banker_id] = book(data, banker_id)
        return book_cache[banker_id]

    @app.get("/api/evals")
    def get_evals() -> list[dict[str, Any]]:
        return eval_runs(settings.runs_dir)

    @app.get("/api/traces/{run_id}")
    def get_trace(run_id: str) -> dict[str, Any]:
        path = find_trace(settings.runs_dir, run_id)
        if path is None:
            raise HTTPException(404, f"No trace '{run_id}'.")
        return trace_payload(Trace.from_dict(json.loads(path.read_text())))

    @app.post("/api/ask")
    def ask(req: AskRequest) -> StreamingResponse:
        check_banker(req.banker_id)
        events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

        def work() -> None:
            try:
                agent = build_agent(
                    settings,
                    data,
                    req.banker_id,
                    llm=llm,
                    model=req.model or None,
                    use_cache=use_cache,
                    grounder=check_grounding,
                    on_event=lambda kind, payload: events.put((kind, payload)),
                )
                history: list[Any] = [t.model_dump() for t in req.history]
                result = agent.run(req.question, history=history)
                result.trace.save(settings.runs_dir / "traces")
                events.put(("done", trace_payload(result.trace)))
            except Exception as exc:  # reported to the page instead of a broken stream
                events.put(("fatal", {"message": str(exc)}))
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()

        def stream() -> Iterator[str]:
            while (item := events.get()) is not None:
                kind, payload = item
                yield f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
