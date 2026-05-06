from __future__ import annotations

import prompts.skills.medical_kb
import prompts.skills.conversation
import prompts.skills.time
import prompts.skills.calculator
import prompts.skills.web_search

from prompts.skills import get_prompts_for_skills

_ROLE = (
    "你是一名谨慎的医疗 AI 助手。"
    "你可以基于医学知识库和工具提供信息检索与解释，但不能替代执业医生做诊断或治疗决策。"
)

_GLOBAL_RULES = (
    "最终回复必须与用户最新一条消息保持同一语言。"
    " 医疗相关回答应保持谨慎、清晰，不得编造医学依据。"
    " 如涉及风险判断、诊断或治疗建议，应提醒用户咨询专业医生。"
)

_DEFAULT_SKILLS = ["medical_kb", "conversation", "time", "calculator"]
_PRIORITY_MAP: dict[str, str] = {
    "medical_kb": "医学知识库问题 → search_medical_kb",
    "conversation": "通用医疗对话、问候、免责声明说明",
    "time": "时间/日期问题 → get_current_time",
    "calculator": "数学表达式 → calculate",
    "web_search": "知识库未覆盖/最新医学进展 → web_search",
}


def build_system_prompt(enabled_skills: list[str] | None = None) -> str:
    skills = enabled_skills or _DEFAULT_SKILLS

    priority_lines = []
    for i, s in enumerate(skills, 1):
        desc = _PRIORITY_MAP.get(s, s)
        priority_lines.append(f"{i}. {desc}")
    priority_section = "工具调用优先级（从高到低）：\n" + "\n".join(priority_lines)

    skill_prompts = get_prompts_for_skills(skills)
    skill_section = "\n\n".join(skill_prompts) if skill_prompts else ""

    parts = [_ROLE, _GLOBAL_RULES, priority_section]
    if skill_section:
        parts.append("以下是每个工具的详细使用规范：\n\n" + skill_section)

    return "\n\n".join(parts)


SYSTEM_PROMPT = build_system_prompt()