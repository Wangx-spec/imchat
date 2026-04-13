from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class SkillDef:
    name: str
    tool_names: list[str]
    prompt: str


_REGISTRY: dict[str, SkillDef] = {}

def register(skill: SkillDef) -> None:
    _REGISTRY[skill.name] = skill

def get_skill(name: str) -> SkillDef | None:
    return _REGISTRY.get(name)

def all_skills() -> list[SkillDef]:
    return list(_REGISTRY.values())

def get_prompts_for_skills(names: list[str]) -> list[str]:
    prompts = []
    for n in names:
        s = _REGISTRY.get(n)
        if s:
            prompts.append(s.prompt)
    return prompts

def get_tool_names_for_skills(names: list[str]) -> set[str]:
    tool_names = set()
    for n in names:
        s = _REGISTRY.get(n)
        if s:
            tool_names.update(s.tool_names)
    return tool_names