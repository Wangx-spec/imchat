from __future__ import annotations

import prompts.skills.knowledge_base  
import prompts.skills.time            
import prompts.skills.calculator      
import prompts.skills.dish_recommend  

from prompts.skills import get_prompts_for_skills

_ROLE = "你是一名有帮助的智能烹饪助手。"

_GLOBAL_RULES = "最终回复必须与用户最新一条消息保持同一语言。"

_DEFAULT_SKILLS = ["knowledge_base", "time", "calculator", "dish_recommend"]

_PRIORITY_MAP: dict[str, str] = {
    "knowledge_base": "文档/菜谱/教程/知识库类问题 → search_knowledge_base",
    "time": "时间/日期问题 → get_current_time",
    "calculator": "数学表达式 → calculate",
    "dish_recommend": "多菜推荐/想吃啥/按条件选菜 → recommend_dishes",
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