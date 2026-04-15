from __future__ import annotations
from dataclasses import dataclass
import logging

logger = logging.getLogger("chat.agent_registry")

@dataclass(frozen=True)
class AgentDef:
    name: str                          # "knowledge", "recommend", "chat"
    description: str                   # 给 Supervisor 看的能力描述
    skills: list[str]                  # 该 Agent 拥有的 skill 名称
    model_override: str | None = None  # 可选：给该 Agent 用不同的 LLM

_AGENT_REGISTRY: dict[str, AgentDef] = {}

def register_agent(agent: AgentDef) -> None:
    _AGENT_REGISTRY[agent.name] = agent
    logger.info(
        "[AGENT_REGISTER] name=%s skills=%s",
        agent.name,
        agent.skills,
    )

def get_agent(name: str) -> AgentDef | None:
    return _AGENT_REGISTRY.get(name)

def all_agents() -> list[AgentDef]:
    return list(_AGENT_REGISTRY.values())


register_agent(AgentDef(
    name="knowledge",
    description="回答知识库/菜谱/教程类问题，提供具体做法和步骤",
    skills=["knowledge_base"],
))

register_agent(AgentDef(
    name="recommend",
    description="推荐菜品、列菜单、按条件筛选菜品",
    skills=["dish_recommend"],
))

register_agent(AgentDef(
    name="chat",
    description="通用对话：闲聊、时间查询、数学计算等非知识库问题",
    skills=["time", "calculator"],
))