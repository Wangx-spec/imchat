from __future__ import annotations

import json
import logging

from langchain_core.messages import SystemMessage

from agents.agent_registry import all_agents
from graphs.common_nodes import latest_user_message_text

logger = logging.getLogger(__name__)

def build_triage_node(llm, settings):
    specs = [a for a in all_agents() if a.name != "image_analysis"]
    names = [a.name for a in specs]
    desc = "\n".join(f'- "{a.name}": {a.description}' for a in specs)
    threshold = getattr(settings, "triage_confidence_threshold", 0.75)
    prompt = f"""你是医疗多智能体系统的分诊器。
可用专家：
{desc}
请判断用户最新问题应交给哪个专家，并评估复杂度。
只返回 JSON：
{{"agent": "<专家名>", "reasoning": "<理由>", "confidence": 0.0-1.0, "complexity": 0.0-1.0}}
agent 可选值：{json.dumps(names, ensure_ascii=False)}"""

    def triage_node(state) -> dict:
        # 1) 有图片优先走影像分支
        if state.get("has_image") or state.get("had_image"):
            return {
                "route": "image", 
                "triage_reason": "input_has_image", 
                "triage_confidence": 1.0
            }
        
        resp = llm.invoke(
            [SystemMessage(content=prompt),
            *state["messages"]]
        )
        parsed = _safe_json(getattr(resp, "content", "")) or {}

        agent = parsed.get("agent") if parsed.get("agent") in names else names[0]
        confidence = _clamp(parsed.get("confidence", 0.0))
        complexity = _clamp(parsed.get("complexity", 0.0))
        reason = str(parsed.get("reasoning", "") or "no_reason")

        # 2) 复杂度高 → swarm 并行；否则 single
        if complexity >= getattr(settings, "complexity_threshold", 0.6):
            route = "swarm"
        else:
            route = "single"
        
        # 3) 低置信兜底：交给consultation 单跑
        if confidence < threshold and route == "single":
            agent = "consultation"
            reason = f"low_confidance_fallback:{reason}"
        
        logger.info("[TRIAGE] route=%s agent=%s conf=%.2f cplx=%.2f", route, agent, confidence, complexity)
        return {
            "route": route,
            "target_agent": agent,
            "triage_reason": reason,
            "triage_confidence": confidence,
        }
    
    return triage_node

def _safe_json(text: str) -> dict | None:
    raw = (text or "").strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(raw[s : e + 1])
    except Exception:
        return None

def _clamp(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0



