from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt



from agents.domain_agents import build_domain_agents
from agents.guardrails import LocalGuardrails
from graphs.common_nodes import (
    build_image_caption_node,
    build_image_input_guardrail_node,
    build_input_guardrail_node,
    build_output_guardrail_node,
    latest_user_message_text
)

from graphs.lead_agent import build_decompose_node, build_synthesize_node
from memory.mem0_client import build_mem0_client
from graphs.triage import build_triage_node
from graphs.swarm_state import SwarmState
from llms.openai_chat import build_openai_chat_model
from llms.qwen_vl import build_qwen_vl_client

logger = logging.getLogger(__name__)


def build_hitl_node(settings):
    def hitl_node(state) -> dict:
        if (not settings.hitl_enabled) or (not state.get("needs_human_validation")):
            return {}

        action = (getattr(settings, "hitl_action", "interrupt") or "interrupt").strip().lower()

        if action == "off":
            return {}
        
        if action == "warn":
            # 仅标记/提示，不中断
            note = "该回答涉及影像/诊断倾向，建议人工复核。"
            return {
                "hitl_decision": "warn",
                "messages": [AIMessage(content=note)],
            }
        
        resume_payload = interrupt({
            "type": "human_validation",
            "reason": state.get("hitl_reason") or "needs_human_validation",
            "draft_answer": getattr(state.get("messages", [])[-1], "content", ""),
        })

        # resume 后回填决策
        if isinstance(resume_payload, dict):
            decision = str(resume_payload.get("decision") or "approve")
            note = (resume_payload.get("note") or "").strip()
        else:
            decision = str(resume_payload or "approve")
            note = ""
        out = {"hitl_decision": decision, "hitl_note": note}
        if note:
            out["messages"] = [AIMessage(content=f"[人工复核] {note}")]
        return out

    return hitl_node

def build_swarm_graph(settings, checkpointer=None):
    llm = build_openai_chat_model(settings)
    vlm = build_qwen_vl_client(settings)
    agents = build_domain_agents(settings)
    mem0_client = build_mem0_client(settings)

    def worker_node(state) -> dict:
        # state 里带单个 subtask（通过 Send 注入）
        task = state["__task__"]
        agent = agents[task["agent"]]
        result = agent.invoke({"messages": [HumanMessage(content=task["query"])]})
        answer = result["messages"][-1].content
        return {"contributions": [{"agent": task["agent"], "answer": answer}]}
    
    def single_node(state) -> dict:
        agent = agents[state.get("target_agent") or "consultation"]
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}
    
    def image_node(state) -> dict:

        captions = state.get("image_captions", [])
        ok_caps = [c for c in captions if c.get("ok")]
        if not ok_caps:
            return {
                "messages": [AIMessage(content="未能获取到可分析的图片，请重新上传清晰的图片。")],
            }


        lines = ["【图像分析结果】"]
        needs_hitl = False
        for i, c in enumerate(ok_caps, start=1):
            lines.append(
                f"\n图片[{i}] 类型={c.get('image_type', 'general')} "
                f"医学相关={'是' if c.get('is_medical') else '否'}\n"
                f"可见内容：{str(c.get('caption', '')).strip() or '未能生成有效描述。'}"
            )
            if c.get("is_diagnostic_request"):
                needs_hitl = True
        if needs_hitl:
            lines.append("\n提醒：以上仅为图像可见内容分析，不能替代医生诊断，请咨询专业医生复核。")
        return {
            "messages": [AIMessage(content="\n".join(lines))],
            "needs_human_validation": needs_hitl,
            "hitl_reason": "image_diagnostic_request" if needs_hitl else None,
        }

        
    def fan_out(state):
        return [Send("worker", {**state, "__task__": t}) for t in state["subtasks"]]

    def _user_id_from_config(config: dict | None) -> str:
        conf = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
        thread_id = str(conf.get("thread_id") or "").strip()
        return thread_id or "anonymous"

    def recall_node(state, config=None) -> dict:
        if mem0_client is None:
            return {}
        user_id = _user_id_from_config(config)
        query = latest_user_message_text(state)
        if not query:
            return {}
        memories = mem0_client.recall(user_id=user_id, query=query, limit=5)
        if not memories:
            return {}
        mem_block = "\n".join([f"- {m}" for m in memories])
        return {
            "messages": [SystemMessage(content=f"用户长期记忆（供参考）：\n{mem_block}")]
        }

    def persist_node(state, config=None) -> dict:
        if mem0_client is None:
            return {}
        user_id = _user_id_from_config(config)
        messages = state.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return {}
        user_text = ""
        ai_text = ""
        for msg in reversed(messages):
            if not user_text and getattr(msg, "type", "") == "human":
                user_text = str(getattr(msg, "content", "")).strip()
            if not ai_text and getattr(msg, "type", "") == "ai":
                ai_text = str(getattr(msg, "content", "")).strip()
            if user_text and ai_text:
                break
        if not user_text or not ai_text:
            return {}
        mem0_client.remember(
            user_id=user_id,
            messages=[
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": ai_text},
            ],
        )
        return {}
    guardrails_on = settings.guardrails_enabled
    builder = StateGraph(SwarmState)

    builder.add_node("image_input_guardrail", build_image_input_guardrail_node(settings))
    builder.add_node("image_caption", build_image_caption_node(vlm))
    builder.add_node("triage", build_triage_node(llm, settings))
    builder.add_node("decompose", build_decompose_node(llm, settings))
    builder.add_node("worker", worker_node)
    builder.add_node("synthesize", build_synthesize_node(llm))

    builder.add_node("hitl_node", build_hitl_node(settings))
    if mem0_client is not None:
        builder.add_node("recall_node", recall_node)
        builder.add_node("persist_node", persist_node)
    
    builder.add_node("single", single_node)
    builder.add_node("image", image_node)
    
    
    
    if guardrails_on:
        g = LocalGuardrails(llm, settings)
        builder.add_node("input_guardrail", build_input_guardrail_node(g))
        builder.add_node("output_guardrail", build_output_guardrail_node(g))
    
    builder.add_edge(START, "image_input_guardrail")
    builder.add_conditional_edges(
        "image_input_guardrail",
        lambda s: "blocked" if s.get("blocked") else "caption",
        {"blocked": END, "caption": "image_caption"},
    )
    if guardrails_on:
        builder.add_edge("image_caption", "input_guardrail")
        if mem0_client is not None:
            builder.add_conditional_edges(
                "input_guardrail",
                lambda s: "blocked" if s.get("blocked") else "recall",
                {"blocked": END, "recall": "recall_node"},
            )
            builder.add_edge("recall_node", "triage")
        else:
            builder.add_conditional_edges(
                "input_guardrail",
                lambda s: "blocked" if s.get("blocked") else "triage",
                {"blocked": END, "triage": "triage"},
            )
    else:
        if mem0_client is not None:
            builder.add_edge("image_caption", "recall_node")
            builder.add_edge("recall_node", "triage")
        else:
            builder.add_edge("image_caption", "triage")

    builder.add_conditional_edges(
        "triage",
        lambda s: s.get("route", "single"),
        {"image": "image", "single": "single", "swarm": "decompose"},
    )
    builder.add_conditional_edges("decompose", fan_out, ["worker"])
    builder.add_edge("worker", "synthesize")
    builder.add_edge("single", "hitl_node")
    builder.add_edge("image", "hitl_node")
    builder.add_edge("synthesize", "hitl_node")

    # 末端串联：hitl -> (output_guardrail) -> (persist_node) -> END
    # 保证 persist_node 写回的是经 guardrail 复核后的最终可发送答案
    tail = ["hitl_node"]
    if guardrails_on:
        tail.append("output_guardrail")
    if mem0_client is not None:
        tail.append("persist_node")
    tail.append(END)
    for src, dst in zip(tail, tail[1:]):
        builder.add_edge(src, dst)

    return builder.compile(checkpointer=checkpointer)
