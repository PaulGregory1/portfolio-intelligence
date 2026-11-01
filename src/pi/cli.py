"""Command-line entry point: `pi --help`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.table import Table

from pi.config import get_settings

if TYPE_CHECKING:
    from pi.agent.loop import Agent, AgentResult

app = typer.Typer(
    help="Portfolio Intelligence: ask questions about client portfolios.", no_args_is_help=True
)
data_app = typer.Typer(help="Generate and inspect the synthetic book.", no_args_is_help=True)
app.add_typer(data_app, name="data")
console = Console()


@app.callback()
def main() -> None:
    """Portfolio analytics agent with grounding checks and an evaluation harness."""


@data_app.command("generate")
def data_generate(
    seed: int = typer.Option(42, help="Random seed: same seed, same database."),
    offline: bool = typer.Option(False, help="Use synthetic prices instead of Yahoo Finance."),
) -> None:
    """Build data/portfolio.duckdb: universe, prices, FX, clients and transactions."""
    from pi.data.generate import generate
    from pi.data.store import write_database

    settings = get_settings()
    with console.status("Generating data..."):
        data = generate(
            seed, settings.data_end_date, offline=offline, cache_dir=settings.data_dir / "raw"
        )
        write_database(data, settings.db_path)
    table = Table(title=f"Database written to {settings.db_path}")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name, frame in data.tables().items():
        table.add_row(name, f"{len(frame):,}")
    console.print(table)
    n_syn = len(data.synthetic_tickers)
    console.print(f"Synthetic price series: {n_syn}/{len(data.instruments)} instruments.")


@app.command()
def models(
    filter: str = typer.Option("", "--filter", "-f", help="Only show ids containing this text."),
) -> None:
    """List the models served by the configured LLM endpoint (GET /models)."""
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key.get_secret_value())
    ids = sorted(m.id for m in client.models.list())
    for model_id in ids:
        if filter.lower() in model_id.lower():
            marker = "  <- LLM_MODEL" if model_id == settings.llm_model else ""
            console.print(f"{model_id}{marker}")


@app.command()
def mcp(
    banker: str = typer.Option(..., "--banker", "-b", help="Banker whose book is exposed."),
) -> None:
    """Serve the portfolio tools over MCP (stdio) for the given banker."""
    from pi.data.store import PortfolioData
    from pi.mcp_server import build_server

    build_server(PortfolioData.load(get_settings().db_path), banker).run("stdio")


def _event_printer(verbose: bool) -> Callable[[str, dict[str, Any]], None]:
    def on_event(kind: str, payload: dict[str, Any]) -> None:
        if kind == "tool_call":
            args = payload["arguments"] or "{}"
            console.print(f"  [cyan]→ {payload['name']}[/cyan] [dim]{args}[/dim]")
        elif kind == "tool_result":
            if payload["ok"]:
                console.print(f"    [green]✓[/green] [dim]{payload['duration_ms']:.0f} ms[/dim]")
            else:
                console.print(f"    [red]✗ {payload['error_code']}[/red]")
        elif kind == "grounding_retry":
            numbers = payload["numbers"]
            console.print(f"  [yellow]⚠ ungrounded numbers {numbers}, asking again[/yellow]")
        elif kind == "llm" and verbose:
            it, latency = payload["iteration"], payload["latency_s"]
            console.print(f"  [dim]model call {it}: {latency:.1f} s[/dim]")

    return on_event


def _make_agent(
    banker: str, model: str | None, cache: bool, verbose: bool, grounding: bool = True
) -> Agent:
    from pi.agent.factory import build_agent
    from pi.data.store import PortfolioData

    settings = get_settings()
    grounder = None
    if grounding:
        from pi.agent.grounding import check_grounding

        grounder = check_grounding
    return build_agent(
        settings,
        PortfolioData.load(settings.db_path),
        banker,
        model=model,
        use_cache=cache,
        grounder=grounder,
        on_event=_event_printer(verbose),
    )


def _print_result(result: AgentResult) -> None:
    from rich.markdown import Markdown
    from rich.panel import Panel

    trace = result.trace
    console.print(Panel(Markdown(result.answer), title="Answer", border_style="blue"))
    grounding = trace.grounding or {}
    rate = grounding.get("grounding_rate")
    rate_txt = f"grounding {rate:.0%}" if rate is not None else "no numbers"
    console.print(
        f"[dim]{trace.status} · {trace.iterations} model calls · {len(trace.tool_steps)} tool "
        f"calls · {trace.prompt_tokens + trace.completion_tokens} tokens · "
        f"{trace.wall_time_s:.1f} s · {rate_txt}[/dim]"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question in natural language."),
    banker: str = typer.Option(..., "--banker", "-b", help="Banker id, e.g. B1."),
    model: str | None = typer.Option(None, help="Override LLM_MODEL."),
    cache: bool = typer.Option(False, help="Use the on-disk LLM response cache."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_grounding: bool = typer.Option(False, help="Disable the grounding check."),
) -> None:
    """Ask one question and print the answer, the tool calls and run statistics."""
    agent = _make_agent(banker, model, cache, verbose, grounding=not no_grounding)
    result = agent.run(question)
    path = result.trace.save(get_settings().runs_dir / "traces")
    _print_result(result)
    console.print(f"[dim]trace: {path}[/dim]")


@app.command()
def chat(
    banker: str = typer.Option(..., "--banker", "-b", help="Banker id, e.g. B1."),
    model: str | None = typer.Option(None, help="Override LLM_MODEL."),
    cache: bool = typer.Option(False, help="Use the on-disk LLM response cache."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Interactive session. Tool calls are shown as they happen; type 'exit' to quit."""
    agent = _make_agent(banker, model, cache, verbose)
    settings = get_settings()
    console.print(
        f"[bold]Portfolio Intelligence[/bold] · banker {banker} · model {agent.llm.model}\n"
        "[dim]Ask about your clients' portfolios. 'exit' to quit, 'reset' to clear history.[/dim]"
    )
    history: list[dict[str, Any]] = []
    while True:
        try:
            question = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "reset":
            history.clear()
            continue
        if not question:
            continue
        agent.ctx.accessed_clients.clear()
        result = agent.run(question, history=history)
        result.trace.save(settings.runs_dir / "traces")
        _print_result(result)
        # Keep the conversation, but not the raw tool payloads, to bound the context size.
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result.answer},
        ]


eval_app = typer.Typer(help="Evaluate models on the generated question set.")
app.add_typer(eval_app, name="eval")
DEFAULT_DATASET = "evals/dataset_v1.jsonl"


@eval_app.callback(invoke_without_command=True)
def eval_run(
    ctx: typer.Context,
    model: str | None = typer.Option(None, help="Model slug, default LLM_MODEL."),
    base_url: str | None = typer.Option(None, help="Endpoint, default LLM_BASE_URL."),
    dataset: Path = typer.Option(Path(DEFAULT_DATASET), help="Dataset (jsonl)."),
    concurrency: int = typer.Option(2, help="Questions evaluated in parallel."),
    cache: bool = typer.Option(False, help="Use the on-disk LLM response cache."),
    limit: int | None = typer.Option(None, help="Only the first N items (after filtering)."),
    category: list[str] | None = typer.Option(None, help="Only these categories."),
    out: Path | None = typer.Option(None, help="Output directory; an existing one is resumed."),
    retry_errors: bool = typer.Option(
        False, help="When resuming, rerun items that hit API errors."
    ),
) -> None:
    """Run the evaluation (`pi eval --model ...`). Subcommands: dataset, compare."""
    if ctx.invoked_subcommand is not None:
        return
    from pi.data.store import PortfolioData
    from pi.evals.dataset import load_dataset
    from pi.evals.report import fmt
    from pi.evals.runner import RunConfig, default_out_dir, run_eval
    from pi.llm.client import LLMClient

    settings = get_settings()
    model_name = model or settings.llm_model
    items = load_dataset(dataset)
    if category:
        items = [i for i in items if i.category in category]
    if limit:
        items = items[:limit]
    config = RunConfig(
        model=model_name,
        base_url=base_url or settings.llm_base_url,
        dataset=dataset,
        out_dir=out or default_out_dir(settings.runs_dir, model_name),
        concurrency=concurrency,
        use_cache=cache,
        price_in=settings.llm_price_input_per_mtok,
        price_out=settings.llm_price_output_per_mtok,
        retry_errors=retry_errors,
    )
    client = LLMClient.from_settings(settings, model=model_name, base_url=base_url, use_cache=cache)
    data = PortfolioData.load(settings.db_path)
    console.print(f"Evaluating [bold]{model_name}[/bold] on {len(items)} items → {config.out_dir}")
    counter = {"n": 0}

    def progress(row: dict[str, Any]) -> None:
        counter["n"] += 1
        mark = "[green]✓[/green]" if row["success"] else "[red]✗[/red]"
        console.print(
            f"{mark} {row['id']:<18} {row['status']:<14} {row['latency_s']:5.1f}s "
            f"tools={','.join(row['tools_called']) or '-'}"
        )

    summary = run_eval(items, data, config, lambda _: client, progress)
    o = summary["overall"]
    console.print(
        f"\nsuccess {fmt(o['success_rate'], 'pct')} · numbers {fmt(o['numeric_accuracy'], 'pct')}"
        f" · grounding {fmt(o['grounding_rate'], 'pct')} · access violations "
        f"{o['access_violations']} · p50 {fmt(o['latency_p50_s'], 'f1')} s"
    )
    console.print(f"Report: {config.out_dir / 'report.md'}")


@eval_app.command("dataset")
def eval_dataset(
    out: Path = typer.Option(Path(DEFAULT_DATASET), help="Where to write the jsonl file."),
    seed: int = typer.Option(7, help="Seed of the template sampler."),
) -> None:
    """Regenerate the question set; ground truth is computed by the analytics engine."""
    from collections import Counter

    from pi.data.store import PortfolioData
    from pi.evals.dataset import build_dataset, save_dataset

    items = build_dataset(PortfolioData.load(get_settings().db_path), seed)
    save_dataset(items, out)
    counts = Counter(i.category for i in items)
    console.print(f"{len(items)} items written to {out}: {dict(counts)}")


@eval_app.command("compare")
def eval_compare(
    runs: list[Path] = typer.Argument(..., help="Run directories (runs/evals/...)."),
    out: Path | None = typer.Option(None, help="Also write the table to this file."),
) -> None:
    """Markdown comparison table of several evaluation runs."""
    from pi.evals.report import compare_runs

    table = compare_runs(runs)
    console.print(table, markup=False)
    if out:
        out.write_text(table, encoding="utf-8")


@eval_app.command("rescore")
def eval_rescore(
    runs: list[Path] = typer.Argument(..., help="Run directories to rescore."),
    dataset: Path = typer.Option(Path(DEFAULT_DATASET), help="Dataset used by the runs."),
) -> None:
    """Recompute scores, grounding and reports of finished runs from their saved traces."""
    from pi.data.store import PortfolioData
    from pi.evals.dataset import load_dataset
    from pi.evals.report import fmt
    from pi.evals.runner import rescore

    settings = get_settings()
    items = load_dataset(dataset)
    data = PortfolioData.load(settings.db_path)
    for run in runs:
        summary = rescore(
            items, data, run, settings.llm_price_input_per_mtok, settings.llm_price_output_per_mtok
        )
        console.print(f"{run}: success {fmt(summary['overall']['success_rate'], 'pct')}")
