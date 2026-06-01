from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, List

@dataclass
class SkillParameter:
    """Skill 参数定义"""
    name: str
    type: str  # "string", "number", "integer", "boolean", "object", "array"
    description: str
    required: bool = False
    enum: Optional[List[str]] = None

@dataclass
class Skill:
    name: str
    description: str
    function: Callable[..., Any]
    parameters: list[SkillParameter]

    def to_openai_tool(self) -> dict[str, Any]:
        properties = {}
        required = []
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
                
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


