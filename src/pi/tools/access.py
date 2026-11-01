"""Execution context injected by the harness, and the banker -> client access check.

The banker id never comes from the model: it is set once by whoever runs the agent (CLI
option, MCP server option, evaluation item) and carried in `ToolContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pi.data.store import Client, PortfolioData

if TYPE_CHECKING:
    from pi.screening.search import Searcher


class ToolFailure(Exception):
    """An error reported back to the model as a structured {code, message} payload."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ToolContext:
    banker_id: str
    data: PortfolioData
    searcher: Searcher | None = None
    # Every client id the tools resolved, used by the evaluation to count access violations.
    accessed_clients: list[str] = field(default_factory=list)


def resolve_client(ctx: ToolContext, ref: str) -> Client:
    """Find a client by id or name, and check that it belongs to the current banker."""
    key = ref.strip().lower()
    match = next(
        (c for c in ctx.data.clients.values() if key in (c.client_id.lower(), c.name.lower())),
        None,
    )
    if match is None:
        raise ToolFailure("NOT_FOUND", f"No client '{ref}'. Use list_clients to see your clients.")
    if match.banker_id != ctx.banker_id:
        raise ToolFailure(
            "ACCESS_DENIED", f"Client '{ref}' is not in the book of banker {ctx.banker_id}."
        )
    ctx.accessed_clients.append(match.client_id)
    return match
