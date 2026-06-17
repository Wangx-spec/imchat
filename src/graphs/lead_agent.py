from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agents.agent_registry import all_agents

logger = logging.getLogger(__name__)


def build_decompose_node(llm, settings):
    names = [a.name for a in all_agents() if a.name != "image_analysis"]
    max_workers = getattr(settings, "swarm_max_workers", 3)
    prompt = f"""把用户的复杂医疗问题拆成最多 {max_workers} 个子任务，每个交给一个专家。
专家可选：{json.dumps(names, ensure_ascii=False)}
只返回 JSON 数组：[{{"agent": "...", "query": "..."}}]"""

    def decompose_node(state) -> dict:
        resp = llm.invoke(
            [SystemMessage(content=prompt),
            *state["messages"]]
        )
        arr = _safe_list(getattr(resp, "content", "")) or []
        subtasks = []
        for i, item in enumerate(arr[:max_workers]):
            agent = item.get("agent")
            if agent in names and item.get("query"):
                subtasks.append({
                    "id": f"t{i}",
                    "agent": agent,
                    "query": item["query"],
                })
        if not subtasks:  # 兜底：单专家
            subtasks = [{"id": "t0", "agent": "consultation",
                         "query": _latest_user(state)}]
        logger.info("[DECOMPOSE] subtasks=%d", len(subtasks))
        return {"subtasks": subtasks}

    return decompose_node

def build_synthesize_node(llm):
    def synthesize_node(state) -> dict:
        contributions = state.get("contributions", [])
        merged = "\n\n".join(
            f"[{c.get('agent')}]\n{c.get('answer', '')}" for c in contributions
        )

        resp = llm.invoke([
            SystemMessage(content="综合各专家结论，输出一份连贯、谨慎、非确诊的中文回答。"),
            HumanMessage(content=f"用户问题：{_latest_user(state)}\n\n各专家结论：\n{merged}"),
        ])

        text = getattr(resp, "content", "") or merged
        return {
            "final_answer": text,
            "messages": [AIMessage(content=text)]
        }

    return synthesize_node

def _latest_user(state) -> str:
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""

def _safe_list(text: str):
    raw = (text or "").strip()
    s, e = raw.find("["), raw.rfind("]")
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(raw[s : e + 1])
    except Exception:
        return None