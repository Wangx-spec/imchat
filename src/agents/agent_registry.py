from __future__ import annotations
from dataclasses import dataclass
import logging

logger = logging.getLogger("chat.agent_registry")

@dataclass(frozen=True)
class LLMSpec:
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None

@dataclass(frozen=True)
class AgentDef:
    name: str                          # "knowledge", "recommend", "chat"
    description: str                   # 给 Supervisor 看的能力描述
    skills: list[str]                  # 该 Agent 拥有的 skill 名称
    model_override: str | None = None  # 可选：给该 Agent 用不同的 LLM
    llm_override: LLMSpec | None = None # 新字段

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
    name="medical_kb",
    description="回答医学知识库内已有的问题（脑肿瘤、胸片影像、皮肤病变、糖尿病等）",
    skills=["medical_kb"],
))

register_agent(AgentDef(
    name="conversation",
    description="通用医疗对话：问候、澄清、免责声明说明、非知识库的一般性沟通",
    skills=["conversation", "time", "calculator"],
))

register_agent(AgentDef(
    name="web_search",
    description="查询知识库未覆盖的医学问题、最新指南、近期研究进展和时效性医疗信息",
    skills=["web_search"],
    llm_override=LLMSpec(model="qwen-plus", temperature=0.1, top_p=0.8),
))