"""Command-line entry point: `pi --help`."""

from __future__ import annotations

import typer

app = typer.Typer(
    help="Portfolio Intelligence: ask questions about client portfolios.", no_args_is_help=True
)


@app.callback()
def main() -> None:
    """Portfolio analytics agent with grounding checks and an evaluation harness."""
