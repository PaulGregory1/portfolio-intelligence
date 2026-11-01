"""Command-line entry point: `pi --help`."""

from __future__ import annotations

from collections.abc import Callable
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
