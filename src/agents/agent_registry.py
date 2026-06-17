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
class AgentSpec:
    name: str                          # "knowledge", "recommend", "chat"
    description: str                   # 给 Supervisor 看的能力描述
    skill_names: list[str]                  # 该 Agent 拥有的 skill 名称
    system_prompt: str                   # 该 Agent 的系统提示词
    llm_override: LLMSpec | None = None # 可选：给该 Agent 用不同的 LLM


_AGENT_REGISTRY: dict[str, AgentSpec] = {}



def register_agent(agent: AgentSpec) -> None:
    _AGENT_REGISTRY[agent.name] = agent
    logger.info(
        "[AGENT_REGISTER] name=%s skill_names=%s",
        agent.name,
        agent.skill_names,
    )

def get_agent(name: str) -> AgentSpec | None:
    return _AGENT_REGISTRY.get(name)

def all_agents() -> list[AgentSpec]:
    return list(_AGENT_REGISTRY.values())



register_agent(AgentSpec(
    name="consultation",
    description="面向用户的一般医学咨询、症状澄清、健康建议与风险提醒。",
    skill_names=[
        "search_knowledge",
        "assess_risk",
        "analyze_symptoms",
        "clinical_guideline",
        "web_search",
    ],
    system_prompt="""你是医学咨询 Agent。你的任务是与用户进行清晰、谨慎、友好的医学沟通。
你可以调用工具检索知识库、分析症状、评估风险或查询指南。
不要替代医生诊断。涉及严重症状时应建议及时就医。"""
))

register_agent(AgentSpec(
    name="diagnostic",
    description="面向诊断辅助的 Agent，负责症状模式分析、风险评估、疾病编码、指南检索与图像辅助分析。",
    skill_names=[
        "analyze_symptoms",
        "assess_risk",
        "disease_code",
        "clinical_guideline",
        "search_knowledge",
        "analyze_image",
    ],
    system_prompt="""你是医学诊断辅助 Agent。你的任务是帮助用户整理症状、风险、可能方向和进一步检查建议。
你不能给出确定诊断或处方。涉及图片时只能描述可见内容和提示风险，最终诊断必须由医生确认。"""
))

register_agent(AgentSpec(
    name="research",
    description="面向医学研究、最新资料、指南和证据补充的 Agent。",
    skill_names=[
        "deep_research",
        "web_search",
        "search_knowledge",
        "clinical_guideline",
    ],
    system_prompt="""你是医学研究 Agent。你的任务是检索、比较和总结医学资料、指南、研究进展和证据。
优先说明信息来源、证据强弱和不确定性。"""
))

register_agent(AgentSpec(
    name="image_analysis",
    description="面向用户上传图片的医学图像分析 Agent，基于视觉大模型进行描述与风险提示。",
    skill_names=["analyze_image"],
    system_prompt="""你是医学图像分析 Agent。你只能描述图片中可见内容、图片类型、医学相关性和风险提示。
不要输出临床确诊。若用户要求诊断，应明确建议由专业医生复核。"""
))