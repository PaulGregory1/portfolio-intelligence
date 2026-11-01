"""Command-line entry point: `pi --help`."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from pi.config import get_settings

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
