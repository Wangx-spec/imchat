from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState


class SwarmState(MessagesState):
    # 路由
    route: str | None
    target_agent: str | None
    triage_reason: str | None
    triage_confidence: float

    # 任务分解 / 并行
    subtasks: list[dict]
    contributions: Annotated[list, operator.add]
    final_answer: str | None
    evidence_answered: bool
    evidence_source: str | None

    # 多模态（复用 common_nodes 已有字段）
    attachments: list[dict]
    had_image: bool
    image_captions: list[dict]
    has_image: bool
    image_type: str | None

    # HITL
    needs_human_validation: bool
    hitl_reason: str | None
    hitl_decision: str | None
    hitl_note: str | None

    blocked: bool