import json
import logging
import re
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.agent_registry import all_agents, LLMSpec
from agents.guardrails import LocalGuardrails
from agents.guardrails.image_guardrails import ImageGuardrailResult, check_image_attachments
from llms.openai_chat import build_openai_chat_model, build_openai_chat_model_with_override
from llms.qwen_vl import build_qwen_vl_client
from langgraph.prebuilt import create_react_agent
from actions.basic_tools import get_actions
from actions.knowledge_base_tools import search_medical_kb
from actions.web_search_tools import web_search
from prompts.system_prompts import build_system_prompt


logger = logging.getLogger(__name__)

_INSUFFICIENT_PATTERNS = [
    # 中文
    "信息不足", "证据不足", "资料不足", "无法回答", "无法确定", "无法判断",
    "没有足够信息", "缺乏证据", "未检索到相关内容", "知识库未覆盖",
    # 英文
    "insufficient information", "not enough information", "lack of evidence",
    "cannot answer", "can't answer", "unable to determine", "cannot determine",
]

_TIME_SENSITIVE_PATTERNS = [
    "最新", "近期", "最近", "今年", "去年", "进展", "更新", "新版", "新指南",
    "latest", "recent", "update", "updated", "guideline", "guidelines",
    "consensus", "2024", "2025", "2026",
]

class MultiAgentState(MessagesState):
    next: str | None
    blocked: bool
    supervisor_reason: str | None
    supervisor_confidence: float
    handoff_to: str | None
    handoff_reason: str | None
    # 多模态
    attachments: list[dict]   # [{image_id, image_path, public_url, mime, size}]
    had_image: bool
    image_captions: list[dict]

def _contains_insufficient_info(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    t = re.sub(r"\s+", " ", t)
    return any(p in t for p in _INSUFFICIENT_PATTERNS)

def _format_captions_block(captions: list[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(captions, start=1):
        if not c.get("ok"):
            lines.append(f"图片[{i}] 处理失败：{c.get('error') or 'unknown'}")
            continue

        lines.append(
            f"图片[{i}] 类型={c.get('image_type', 'general')} "
            f"is_medical={c.get('is_medical', False)}\n"
            f"  摘要：{str(c.get('caption', '')).strip()}"
        )

    return "\n".join(lines) if lines else "（无）"


def _is_time_sensitive_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    t = re.sub(r"\s+", " ", t)
    return any(p in t for p in _TIME_SENSITIVE_PATTERNS)

def _preview_latest_user_message(state: MessagesState, max_chars: int = 80) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            content = getattr(msg, "content", "")
            text = content if isinstance(content, str) else str(content)
            text = text.strip().replace("\n", " ")
            if len(text) > max_chars:
                return text[:max_chars] + "..."
            return text
    return ""

def _latest_user_message_text(state: MessagesState) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""

def _latest_ai_message(state: MessagesState):
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "ai":
            return msg
    return None

def _build_input_guardrail_node(guardrails: LocalGuardrails):
    def input_guardrail_node(state: MultiAgentState) -> dict:
        user_text = _latest_user_message_text(state)
        logger.info("[INPUT_GUARDRAIL] start latest_user=%r", user_text[:120])

        allowed, message = guardrails.check_input(user_text)
        if allowed:
            logger.info("[INPUT_GUARDRAIL] allowed")
            return {"blocked": False}

        logger.info("[INPUT_GUARDRAIL] blocked reason=%r", message)
        return {
            "blocked": True,
            "messages": [AIMessage(content=message)],
        }

    return input_guardrail_node

def _build_image_input_guardrail_node(settings):
    def node(state: MultiAgentState) -> dict:
        if not state.get("had_image"):
            return {}
        result: ImageGuardrailResult = check_image_attachments(
            attachments=state.get("attachments") or [],
            allowed_mime=settings.multimodal_allowed_mime,
            max_image_bytes=settings.multimodal_max_image_bytes,
            max_images=settings.multimodal_max_images_per_request,
        )
        if result.ok:
            return {}
        return {
            "blocked": True,
            "messages": [AIMessage(content=result.user_message)],
        }
    return node

def _build_image_caption_node(vlm):
    def node(state: MultiAgentState) -> dict:
        if not state.get("had_image"):
            return {}

        attachments = state.get("attachments") or []
        if vlm is None or not attachments:
            return {
                "image_captions": [],
                "messages": [
                    HumanMessage(content="[图片处理] 多模态服务不可用，本轮请改为文字描述。")
                ],
            }

        captions: list[dict] = []
        for att in attachments:
            cap = vlm.summarize_image(image_path=att.get("image_path"))
            captions.append({
                "image_id": att.get("image_id"),
                "image_path": att.get("image_path"),
                "public_url": att.get("public_url"),
                "ok": cap.ok,
                "caption": cap.caption,
                "image_type": cap.image_type,
                "is_medical": cap.is_medical,
                "is_diagnostic_request": cap.is_diagnostic_request,
                "error": cap.error,
            })

        original_text = _latest_user_message_text(state)
        captions_block = _format_captions_block(captions)
        new_text = (
            f"{original_text.strip()}\n\n[随附图片摘要]\n{captions_block}"
            if original_text.strip()
            else f"用户上传了图片，请基于以下图片摘要回答。\n\n[随附图片摘要]\n{captions_block}"
        )

        logger.info(
            "[IMAGE_CAPTION] count=%d ok_count=%d",
            len(captions),
            sum(1 for c in captions if c.get("ok")),
        )

        return {
            "image_captions": captions,
            "messages": [HumanMessage(content=new_text)],
        }

    return node

def _build_output_guardrail_node(guardrails: LocalGuardrails):
    def output_guardrail_node(state: MultiAgentState) -> dict:
        user_text = _latest_user_message_text(state)
        last_ai = _latest_ai_message(state)
        if last_ai is None:
            logger.info("[OUTPUT_GUARDRAIL] no_ai_message_skip")
            return {}

        original_text = getattr(last_ai, "content", "")
        revised_text = guardrails.check_output(original_text, user_text)

        if revised_text == original_text:
            logger.info("[OUTPUT_GUARDRAIL] unchanged")
            return {}

        logger.info("[OUTPUT_GUARDRAIL] revised")
        return {
            "messages": [
                AIMessage(
                    content=revised_text,
                    id=getattr(last_ai, "id", None),  # 关键：尽量复用 id，覆盖原消息
                )
            ]
        }

    return output_guardrail_node

def _extract_latest_json_payload_from_messages(messages, expected_type: str | None = None) -> dict | None:
    for msg in reversed(messages or []):
        content = getattr(msg, "content", None)
        if not isinstance(content, str):
            continue
        text = content.strip()
        if not text.startswith("{"):
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue
        if expected_type and data.get("type") != expected_type:
            continue
        return data
    return None


def _extract_json_object(text: str) -> dict | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return None

def _build_supervisor_node(llm, agent_defs, confidence_threshold: float):
    agent_descriptions = "\n".join(
        f'- "{a.name}": {a.description}'
        for a in agent_defs
    )
    agent_names = [a.name for a in agent_defs]
    fallback_agent = "conversation" if "conversation" in agent_names else agent_names[-1]
    router_prompt = f"""\
你是一个医疗多智能体系统的路由器，负责将用户消息分派给最合适的专家。
可用专家：
{agent_descriptions}
规则：
1. 分析用户最新一条消息的意图
2. 选择最匹配的专家
3. 给出一句简短理由
4. 给出 0.0 到 1.0 之间的置信度
5. 只返回 JSON，不要输出其他文字
返回格式：
{{"agent": "<专家名>", "reason": "<一句话理由>", "confidence": 0.83}}
可选的 agent 值: {json.dumps(agent_names)}
"""
    def supervisor_node(state: MultiAgentState) -> dict:
        latest_user = _preview_latest_user_message(state)
        logger.info("[SUPERVISOR] start latest_user=%r", latest_user)
        messages = [
            SystemMessage(content=router_prompt),
            *state["messages"],
        ]
        response = llm.invoke(messages)
        chosen = fallback_agent
        reason = "fallback"
        confidence = 0.0
        try:
            parsed = _extract_json_object(getattr(response, "content", ""))
            if not isinstance(parsed, dict):
                raise ValueError("router_response_not_json")
            candidate = parsed.get("agent", fallback_agent)
            if candidate in agent_names:
                chosen = candidate
            reason = str(parsed.get("reason", "")).strip() or "no_reason"
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            logger.warning("[SUPERVISOR] parse_failed raw=%r", getattr(response, "content", response))
            if _is_time_sensitive_query(latest_user) and "web_search" in agent_names:
                chosen = "web_search"
                reason = "parse_failed_time_sensitive_fallback"
                confidence = confidence_threshold
        if confidence < confidence_threshold:
            logger.info(
                "[SUPERVISOR] low_confidence fallback=%s chosen=%s confidence=%.3f reason=%r",
                fallback_agent,
                chosen,
                confidence,
                reason,
            )
            return {
                "next": fallback_agent,
                "supervisor_reason": f"low_confidence_fallback: {reason}",
                "supervisor_confidence": confidence,
            }
        logger.info(
            "[SUPERVISOR] routed_to=%s confidence=%.3f reason=%r",
            chosen,
            confidence,
            reason,
        )
        return {
            "next": chosen,
            "supervisor_reason": reason,
            "supervisor_confidence": confidence,
        }
    return supervisor_node


def _resolve_specialist_llm(default_llm, agent_def, settings):
    # 新字段优先，兼容旧字段 model_override
    override = getattr(agent_def, "llm_override", None)
    if override is None:
        model_override = getattr(agent_def, "model_override", None)
        if model_override:
            override = LLMSpec(model=model_override)

    if override is None:
        return default_llm
    return build_openai_chat_model_with_override(settings, override)


def _invoke_tool_payload(tool_obj, tool_name: str, query: str) -> tuple[str, list[dict]]:
    payload_text = tool_obj.invoke({"query": query})
    tool_call = [{
        "id": f"{tool_name}:{abs(hash((tool_name, query))) % 10_000_000}",
        "type": "tool_call",
        "name": tool_name,
        "args": {"query": query},
    }]
    return payload_text, tool_call


def _answer_from_tool_payload(name: str, specialist_llm, prompt: str, user_query: str, payload_text: str) -> str:
    payload = _extract_json_object(payload_text) or {}
    ok = bool(payload.get("ok", False))
    error = str(payload.get("error", "") or "").strip()
    answer = str(payload.get("answer", "") or "").strip()
    results = payload.get("results") or []

    if name == "medical_kb":
        if answer:
            return answer
        if error:
            return f"知识库检索暂时不可用：{error}"
        return "未从知识库检索到可用结果。"

    if not ok or error:
        if error == "tavily_disabled":
            return "本次未获得有效联网结果：联网搜索当前已关闭。若需要最新资料，请先启用 Tavily 再重试。"
        return f"本次未获得有效联网结果。{('错误：' + error) if error else ''}".strip()
    if not results:
        return "本次未检索到足够的联网结果，暂无法提供可靠的实时补充信息。"

    grounded_prompt = (
        prompt
        + "\n\n你已经收到工具返回的 JSON 结果。"
        + " 现在必须严格依据该 JSON 中的 answer/results/error 作答。"
        + " 绝对禁止声称自己已查阅未出现在 JSON 中的指南、期刊、作者、DOI 或网页。"
        + " 如果 JSON 未提供有效结果，必须明确说明未获得有效联网结果。"
        + " 直接输出给用户的最终回答，不要输出工具解释。"
    )
    out = specialist_llm.invoke([
        SystemMessage(content=grounded_prompt),
        HumanMessage(content=f"用户问题：\n{user_query}\n\n工具 JSON：\n{payload_text}"),
    ])
    text = getattr(out, "content", "")
    return text if isinstance(text, str) and text.strip() else answer


def _build_specialist_node(name, default_llm, agent_def, settings):
    tools = get_actions(agent_def.skills)
    prompt = build_system_prompt(agent_def.skills)
    tool_names = [getattr(tool, "name", str(tool)) for tool in tools]
    specialist_llm = _resolve_specialist_llm(default_llm, agent_def, settings)
    llm_override = getattr(agent_def, "llm_override", None)
    if llm_override is None and getattr(agent_def, "model_override", None):
        llm_override = LLMSpec(model=agent_def.model_override)

    logger.info(
        "[SPECIALIST_BUILD] name=%s skills=%s tools=%s llm_override=%s",
        name,
        agent_def.skills,
        tool_names,
        llm_override,
    )
    forced_tool_names = {"medical_kb", "web_search"}
    sub_agent = None
    if name not in forced_tool_names:
        sub_agent = create_react_agent(
            model=specialist_llm,
            tools=tools,
            prompt=prompt,
        )

    def specialist_node(state: MultiAgentState) -> dict:
        latest_user = _preview_latest_user_message(state)
        logger.info("[SPECIALIST_RUN] name=%s start latest_user=%r", name, latest_user)
        current_query = _latest_user_message_text(state)
        if name == "medical_kb":
            payload_text, tool_calls = _invoke_tool_payload(search_medical_kb, "search_medical_kb", current_query)
            final_answer = _answer_from_tool_payload(name, specialist_llm, prompt, current_query, payload_text)
            result = {
                "messages": [
                    AIMessage(content=payload_text, additional_kwargs={"tool_calls": tool_calls}),
                    AIMessage(content=final_answer),
                ]
            }
        elif name == "web_search":
            payload_text, tool_calls = _invoke_tool_payload(web_search, "web_search", current_query)
            final_answer = _answer_from_tool_payload(name, specialist_llm, prompt, current_query, payload_text)
            result = {
                "messages": [
                    AIMessage(content=payload_text, additional_kwargs={"tool_calls": tool_calls}),
                    AIMessage(content=final_answer),
                ]
            }
        else:
            result = sub_agent.invoke({"messages": state["messages"]})
        out: dict = {
            "messages": result["messages"],
            "handoff_to": None,
            "handoff_reason": None,
        }
        if name == "medical_kb":
            payload = _extract_latest_json_payload_from_messages(
                result.get("messages", []),
                expected_type="kb_result",
            )
            if payload:
                debug = payload.get("debug") or {}

                answer_text = str(payload.get("answer", "") or "")
                insufficient_info = _contains_insufficient_info(answer_text)
                is_time_sensitive = _is_time_sensitive_query(current_query)

                low_confidence = bool(debug.get("low_confidence_blocked", False))
                is_confident = bool(debug.get("is_confident", False))
                confidence_score = float(debug.get("confidence_score", 0.0) or 0.0)
                should_handoff = (
                    low_confidence
                    or insufficient_info
                    or (
                        is_time_sensitive
                        and (not is_confident)
                        and confidence_score < 0.45
                    )
                )
                if should_handoff:
                    out["handoff_to"] = "web_search"
                    if insufficient_info and not low_confidence:
                        out["handoff_reason"] = "medical_kb_insufficient_info"
                    elif is_time_sensitive and not low_confidence:
                        out["handoff_reason"] = f"medical_kb_time_sensitive_low_confidence:{confidence_score:.3f}"
                    else:
                        out["handoff_reason"] = f"medical_kb_low_confidence:{confidence_score:.3f}"
                logger.info(
                    "[KB_HANDOFF_CHECK] low_confidence=%s is_confident=%s confidence_score=%.3f insufficient_info=%s is_time_sensitive=%s",
                    low_confidence, is_confident, confidence_score, insufficient_info, is_time_sensitive
                )
        logger.info(
            "[SPECIALIST_RUN] name=%s done message_count=%d handoff_to=%s",
            name,
            len(result.get("messages", [])),
            out.get("handoff_to"),
        )
        return out

    return specialist_node

def build_multi_agent_graph(settings, checkpointer=None):
    llm = build_openai_chat_model(settings)
    vlm = build_qwen_vl_client(settings)
    agent_defs = all_agents()
    agent_names = [a.name for a in agent_defs]
    final_node = "output_guardrail" if settings.guardrails_enabled else END

    logger.info(
        "[MULTI_AGENT_BUILD] agent_names=%s checkpointer=%s guardrails_enabled=%s supervisor_confidence_threshold=%s",
        agent_names,
        type(checkpointer).__name__ if checkpointer is not None else "None",
        getattr(settings, "guardrails_enabled", True),
        getattr(settings, "supervisor_confidence_threshold", 0.6),
    )

    supervisor = _build_supervisor_node(
        llm,
        agent_defs,
        confidence_threshold=getattr(settings, "supervisor_confidence_threshold", 0.6),
    )
    builder = StateGraph(MultiAgentState)

    builder.add_node("image_input_guardrail", _build_image_input_guardrail_node(settings))
    builder.add_node("image_caption", _build_image_caption_node(vlm))
    builder.add_node("supervisor", supervisor)

    for agent_def in agent_defs:
        node = _build_specialist_node(agent_def.name, llm, agent_def, settings)
        builder.add_node(agent_def.name, node)

    if settings.guardrails_enabled:
        guardrails = LocalGuardrails(llm)
        builder.add_node("input_guardrail", _build_input_guardrail_node(guardrails))
        builder.add_node("output_guardrail", _build_output_guardrail_node(guardrails))

    builder.add_edge(START, "image_input_guardrail")
    builder.add_conditional_edges(
        "image_input_guardrail",
        lambda state: "blocked" if state.get("blocked", False) else "caption",
        {
            "blocked": END,
            "caption": "image_caption",
        },
    )

    if settings.guardrails_enabled:
        builder.add_edge("image_caption", "input_guardrail")

        builder.add_conditional_edges(
            "input_guardrail",
            lambda state: "blocked" if state.get("blocked", False) else "supervisor",
            {
                "blocked": END,
                "supervisor": "supervisor",
            },
        )
    else:
        builder.add_edge("image_caption", "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", agent_names[-1]),
        {name: name for name in agent_names},
    )

    if "medical_kb" in agent_names and "web_search" in agent_names:
        builder.add_conditional_edges(
            "medical_kb",
            lambda state: state.get("handoff_to") or "done",
            {
                "web_search": "web_search",
                "done": final_node,
            },
        )
    elif "medical_kb" in agent_names:
        builder.add_edge("medical_kb", final_node)

    for name in agent_names:
        if name == "medical_kb":
            continue
        builder.add_edge(name, final_node)

    if settings.guardrails_enabled:
        builder.add_edge("output_guardrail", END)

    return builder.compile(checkpointer=checkpointer)