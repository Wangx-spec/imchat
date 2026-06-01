from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from .base import Skill

_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def select(self, names: list[str] | None = None) -> list[Skill]:
        if not names:
            return list(self._skills.values())
        return [self._skills[name] for name in names if name in self._skills]

    def to_openai_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        return [skill.to_openai_tool() for skill in self.select(names)]

    def to_react_tools(self, names: list[str] | None = None) -> list[StructuredTool]:
        """Export StructuredTool instances for LangGraph create_react_agent workers."""
        tools: list[StructuredTool] = []
        for skill in self.select(names):
            fields: dict[str, tuple[type, Any]] = {}
            for param in skill.parameters:
                py_type = _TYPE_MAP.get(param.type, str)
                default = ... if param.required else None
                fields[param.name] = (
                    py_type,
                    Field(default, description=param.description),
                )

            args_schema = create_model(
                f"{skill.name.title().replace('_', '')}Args",
                **fields,
            )

            tools.append(
                StructuredTool.from_function(
                    func=skill.function,
                    name=skill.name,
                    description=skill.description,
                    args_schema=args_schema,
                )
            )
        return tools


registry = SkillRegistry()


def get_react_tools(names: list[str] | None = None) -> list[StructuredTool]:
    return registry.to_react_tools(names)


def get_openai_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    return registry.to_openai_tools(names)
