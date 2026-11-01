"""Wire an agent from settings: LLM client, tools, banker context and grounding check."""

from __future__ import annotations

from pi.agent.loop import Agent, AgentConfig, EventHandler, Grounder
from pi.config import Settings
from pi.data.store import PortfolioData
from pi.llm.client import ChatModel, LLMClient
from pi.tools.access import ToolContext
from pi.tools.portfolio_tools import registry


def build_agent(
    settings: Settings,
    data: PortfolioData,
    banker_id: str,
    *,
    llm: ChatModel | None = None,
    model: str | None = None,
    base_url: str | None = None,
    use_cache: bool = False,
    grounder: Grounder | None = None,
    config: AgentConfig | None = None,
    on_event: EventHandler | None = None,
) -> Agent:
    if banker_id not in data.bankers:
        raise ValueError(f"Unknown banker '{banker_id}'. Known: {', '.join(data.bankers)}.")
    client = llm or LLMClient.from_settings(
        settings, model=model, base_url=base_url, use_cache=use_cache
    )
    ctx = ToolContext(banker_id=banker_id, data=data)
    return Agent(client, registry, ctx, config=config, grounder=grounder, on_event=on_event)
