"""Run an agent over the dataset, item by item, with concurrency and resumability.

Each finished item is appended to `results.jsonl` immediately, so an interrupted run can be
resumed by pointing `--out` at the same directory: completed items are skipped.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pi.agent.grounding import check_grounding
from pi.agent.loop import Agent, AgentConfig
from pi.agent.trace import Trace
from pi.data.store import PortfolioData
from pi.evals.dataset import EvalItem
from pi.evals.metrics import aggregate, score_item
from pi.llm.client import ChatModel
from pi.tools.access import ToolContext
from pi.tools.portfolio_tools import registry

LLMFactory = Callable[[EvalItem], ChatModel]


@dataclass
class RunConfig:
    model: str
    base_url: str
    dataset: Path
    out_dir: Path
    concurrency: int = 2
    use_cache: bool = False
    price_in: float = 0.0
    price_out: float = 0.0


def default_out_dir(runs_dir: Path, model: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", model.split("/")[-1])
    return runs_dir / "evals" / f"{datetime.now():%Y%m%d-%H%M%S}_{slug}"


def completed_ids(results: Path) -> set[str]:
    if not results.exists():
        return set()
    ids = set()
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def result_row(item: EvalItem, trace: Trace, data: PortfolioData) -> dict[str, Any]:
    owners = {cid: c.banker_id for cid, c in data.clients.items()}
    score = score_item(item, trace, owners)
    return {
        **score.to_dict(),
        "question": item.question,
        "banker_id": item.banker_id,
        "language": item.language,
        "answer": trace.answer,
        "tools_called": trace.tools_called,
        "expected_tools": item.expected_tools,
        "expected_values": [v.__dict__ for v in item.expected_values],
        "ungrounded": (trace.grounding or {}).get("ungrounded", []),
        "trace_id": trace.run_id,
        "error": trace.error,
    }


def run_item(
    item: EvalItem, llm: ChatModel, data: PortfolioData, traces_dir: Path
) -> dict[str, Any]:
    ctx = ToolContext(banker_id=item.banker_id, data=data)
    agent = Agent(llm, registry, ctx, config=AgentConfig(), grounder=check_grounding)
    result = agent.run(item.question)
    result.trace.save(traces_dir)
    return result_row(item, result.trace, data)


def rescore(
    items: list[EvalItem],
    data: PortfolioData,
    out: Path,
    price_in: float = 0.0,
    price_out: float = 0.0,
) -> dict[str, Any]:
    """Recompute every score of a finished run from its saved traces, without calling a model.

    Used after a fix in the metrics or the grounding check: the answers stay the same.
    """
    by_id = {item.id: item for item in items}
    traces = {}
    for path in (out / "traces").glob("*.json"):
        trace = Trace.from_dict(json.loads(path.read_text(encoding="utf-8")))
        traces[trace.run_id] = trace
    rows = []
    for old in load_results(out / "results.jsonl"):
        trace = traces[old["trace_id"]]
        trace.grounding = check_grounding(trace.answer, trace.grounding_sources()).to_dict()
        rows.append(result_row(by_id[old["id"]], trace, data))
    (out / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    summary = summarise(rows, price_in, price_out)
    summary["model"] = json.loads((out / "config.json").read_text())["model"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    from pi.evals.report import write_report

    write_report(out, summary, rows)
    return summary


def run_eval(
    items: list[EvalItem],
    data: PortfolioData,
    config: RunConfig,
    llm_factory: LLMFactory,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    out = config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = completed_ids(results_path)
    todo = [item for item in items if item.id not in done]
    (out / "config.json").write_text(
        json.dumps(
            {
                "model": config.model,
                "base_url": config.base_url,
                "dataset": str(config.dataset),
                "concurrency": config.concurrency,
                "cache": config.use_cache,
                "items": len(items),
                "resumed_with": len(done),
            },
            indent=2,
        )
    )
    lock = threading.Lock()

    def work(item: EvalItem) -> dict[str, Any]:
        row = run_item(item, llm_factory(item), data, out / "traces")
        with lock, results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = [pool.submit(work, item) for item in todo]
        for future in as_completed(futures):
            row = future.result()
            if progress:
                progress(row)

    wanted = {item.id for item in items}
    rows = [r for r in load_results(results_path) if r["id"] in wanted]
    summary = summarise(rows, config.price_in, config.price_out)
    summary["model"] = config.model
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    from pi.evals.report import write_report

    write_report(out, summary, rows)
    return summary


def load_results(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    # If an item was written twice (e.g. a crash between writes), keep the last one.
    return list({r["id"]: r for r in rows}.values())


def summarise(rows: list[dict[str, Any]], price_in: float, price_out: float) -> dict[str, Any]:
    categories = sorted({r["category"] for r in rows})
    return {
        "overall": aggregate(rows, price_in, price_out),
        "by_category": {
            c: aggregate([r for r in rows if r["category"] == c], price_in, price_out)
            for c in categories
        },
    }
