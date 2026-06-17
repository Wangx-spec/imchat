from __future__ import annotations

import logging
from typing import Any

from langgraph.prebuilt import create_react_agent
from agents.agent_registry import AgentSpec, all_agents
from llms.openai_chat import build_chat_model
from llms.qwen_vl import build_qwen_vl_client
from skills import registry
from skills.runtime import set_vlm_client

logger = logging.getLogger("chat.domain_agents")

def build_domain_agent(settings, spec: AgentSpec) -> Any:
    llm = build_chat_model(settings, spec.llm_override)
    tools = registry.to_react_tools(spec.skill_names)

    logger.info(
        "[DOMAIN_AGENT_BUILD] name=%s skill_names=%s tool_count=%d llm_override=%s",
        spec.name,
        spec.skill_names,
        len(tools),
        spec.llm_override,
    )

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=spec.system_prompt,
    )

def build_domain_agents(settings) -> dict[str, Any]:
    """Build consultation/diagnostic/research ReAct agents from AgentSpec."""
    _bind_runtime_clients(settings)

    agents: dict[str, Any] = {}
    for spec in all_agents():
        agents[spec.name] = build_domain_agent(settings, spec)

    logger.info("[DOMAIN_AGENTS_BUILD] names=%s", list(agents))

    return agents

def _bind_runtime_clients(settings) -> None:
    """Bind clients used by dynamically loaded skill scripts."""
    if not getattr(settings, "image_analysis_enabled", False):
        set_vlm_client(None)
        return
    vlm = build_qwen_vl_client(settings)
    set_vlm_client(vlm)
