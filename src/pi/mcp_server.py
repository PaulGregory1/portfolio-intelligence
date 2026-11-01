"""The same tools, exposed over the Model Context Protocol (stdio).

The banker is fixed when the server starts (`pi mcp --banker B1`), exactly as the agent
harness does: an MCP client can never choose whose book it reads.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from pi.data.store import PortfolioData
from pi.tools.access import ToolContext
from pi.tools.portfolio_tools import registry
from pi.tools.registry import Tool


def _handler(tool: Tool, ctx: ToolContext) -> Callable[..., str]:
    """A function whose signature mirrors the tool's input model, as MCPServer expects."""

    def handler(**kwargs: Any) -> str:
        result = registry.execute(tool.name, kwargs, ctx)
        return json.dumps(result.content, ensure_ascii=False)

    params = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if field.is_required() else field.default,
            # Keep the field descriptions in the JSON schema the MCP client sees.
            annotation=Annotated[field.annotation, Field(description=field.description)],
        )
        for name, field in tool.input_model.model_fields.items()
    ]
    handler.__signature__ = inspect.Signature(params, return_annotation=str)  # type: ignore[attr-defined]
    handler.__annotations__ = {p.name: p.annotation for p in params} | {"return": str}
    handler.__name__ = tool.name
    return handler


def build_server(data: PortfolioData, banker_id: str) -> MCPServer:
    ctx = ToolContext(banker_id=banker_id, data=data)
    server = MCPServer(
        name="portfolio-intelligence",
        instructions=f"Portfolio analytics tools for the clients of banker {banker_id}.",
    )
    for tool in registry:
        server.add_tool(_handler(tool, ctx), name=tool.name, description=tool.description)
    return server
