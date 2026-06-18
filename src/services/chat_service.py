import re
import time
from threading import Lock
from config.logging_setup import setup_logging
from typing import Any, Generator
import logging
import json
from db.messages import save_message
from db.conversations import update_conversation, update_title

from langchain_core.messages import HumanMessage
from langgraph.types import Command
import base64, mimetypes

_INTERNAL_STREAM_NODES = {
    "supervisor",
    "image_input_guardrail",
    "image_caption",
    "input_guardrail",
    "triage",
    "decompose",
    "worker",
}

_FINAL_STREAM_NODES = {
    "single",
    "image",
    "synthesize",
    "output_guardrail",
    "hitl_node"
}
setup_logging()

_KB_CLAIM_PAT = re.compile(r"(根据知识库|知识库中|参考文档|来源：|source_dir|\.md)", re.IGNORECASE)
logger = logging.getLogger("chat.service")

# ---- 模块级单例，由 init() 注入 ----
_agent: Any = None
_settings: Any = None
_runtime: str = ""
_SESSION_LOCKS: dict[str, Lock] = {}


def _chunk_event_for_text(text: str, sent_answer: str) -> tuple[str, str]:
    is_prefix = text.startswith(sent_answer)
    delta = text[len(sent_answer):] if is_prefix else text
    payload: dict[str, Any] = {"text": delta}
    if not is_prefix:
        payload["replace"] = True
    return sse_event("chunk", payload), text


def resume_stream(session_id: str, decision: str, note: str = "") -> Generator[str, None, None]:
    latest_answer = ""
    collected_tool_calls: list[dict] = []
    interrupted = False
    try:
        cmd = Command(resume={"decision": decision, "note": note})
        latest_answer, collected_tool_calls, interrupted = yield from _stream_langgraph_updates(
            session_id=session_id,
            inputs=cmd,
            current_user_message="",  # resume 无新的 human 文本
        )
        if interrupted:
            return
        if latest_answer:
            latest_answer = sanitize_ungrounded_kb_claim(latest_answer, collected_tool_calls)
            yield sse_event("done", {"answer": latest_answer})
    except Exception as exc:
        yield sse_event("error", {"detail": str(exc)})
    finally:
        if latest_answer and not interrupted:
            save_message(session_id, "assistant", latest_answer)
            update_conversation(session_id)

def _stream_langgraph_updates(
    session_id: str,
    inputs: dict,
    current_user_message: str,
) -> Generator[str, None, tuple[str, list[dict], bool]]:
    latest_answer = ""
    sent_answer = ""
    collected_tool_calls: list[dict] = []
    stream_deferred = _runtime in {"langgraph-multi", "langgraph-swarm"}

    timeout_s = float(getattr(_settings, "agent_timeout_s", 60.0) or 60.0)
    start = time.monotonic()
    timed_out = False

    for update in _agent.stream(
        inputs,
        config={"configurable": {"thread_id": session_id}},
        stream_mode="updates",
    ):
        if time.monotonic() - start > timeout_s:
            timed_out = True
            logger.warning(
                "[CHAT_STREAM_TIMEOUT] session=%s elapsed=%.1fs limit=%.1fs",
                session_id,
                time.monotonic() - start,
                timeout_s,
            )
            break

        if not isinstance(update, dict):
            continue

        if "__interrupt__" in update:
            interrupts = update.get("__interrupt__") or []
            first = interrupts[0] if interrupts else None
            payload = getattr(first, "value", first)
            yield sse_event("hitl", {"interrupt": payload})
            return latest_answer, collected_tool_calls, True

        for node_name, node_state in update.items():
            if _should_skip_stream_node(node_name):
                continue
            if not isinstance(node_state, dict):
                continue

            turn_messages = _current_turn_messages(
                node_state.get("messages", []),
                current_user_message,
            )
            for msg in turn_messages:
                _extend_tool_calls_unique(
                    collected_tool_calls,
                    extract_tool_calls_from_message(msg),
                )
                if getattr(msg, "type", "") != "ai":
                    continue

                text = extract_text_from_content(getattr(msg, "content", "")).strip()
                if not text:
                    continue

                latest_answer = text
                if stream_deferred:
                    continue

                if text != sent_answer:
                    event, sent_answer = _chunk_event_for_text(text, sent_answer)
                    yield event

    if timed_out and not latest_answer:
        latest_answer = (
            "处理时间较长，已根据现有信息为你总结：暂未形成完整结论，"
            "建议补充更具体的症状/病史信息，或稍后重试。如症状紧急，请及时就医。"
        )

    if latest_answer and latest_answer != sent_answer:
        if stream_deferred or timed_out:
            event, sent_answer = _chunk_event_for_text(latest_answer, sent_answer)
            yield event

    return latest_answer, collected_tool_calls, False


def _invoke_agent_with_inputs(session_id: str, inputs: dict) -> tuple[str, list[dict]]:
    result = _agent.invoke(inputs, config={"configurable": {"thread_id": session_id}})
    return extract_text_from_result(result), extract_tool_calls(result)

def _attachment_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b}"

def init(agent: Any, settings: Any, runtime: str) -> None:
    """由 app.py 启动时调一次，注入全局依赖。"""
    global _agent, _settings, _runtime
    _agent = agent
    _settings = settings
    _runtime = runtime

def _build_multimodal_input(message: str, attachments: list[dict]) -> dict:
    parts: list[dict] = []
    if message.strip():
        parts.append({"type": "text", "text": message.strip()})
    for att in attachments:
        url = att.get("public_url") or _attachment_to_data_url(att["image_path"])
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return {
        "messages": [HumanMessage(content=parts)],
        "attachments": attachments,
        "had_image": bool(attachments),
    }

def stream_multimodal(
    session_id: str,
    message: str,
    attachments: list[dict],
) -> Generator[str, None, None]:
    note = f"[已附带 {len(attachments)} 张图片]" if attachments else ""
    saved_text = message.strip() or note
    save_message(session_id, "user", (message + ("\n" + note if note else "")).strip() or note)

    latest_answer = ""
    collected_tool_calls: list[dict] = []
    interrupted = False
    inputs = _build_multimodal_input(message, attachments)

    try:
        if _runtime.startswith("langgraph") and _settings.agent_streaming:
            latest_answer, collected_tool_calls, interrupted = yield from _stream_langgraph_updates(
                session_id,
                inputs,
                saved_text,
            )
            if interrupted:
                return
            if not latest_answer:
                latest_answer, collected_tool_calls = _invoke_agent_with_inputs(session_id, inputs)
                latest_answer = _sanitize_echo_answer(latest_answer, message)
                yield sse_event("chunk", {"text": latest_answer})
        else:
            latest_answer, collected_tool_calls = _invoke_agent_with_inputs(session_id, inputs)
            latest_answer = _sanitize_echo_answer(latest_answer, message)
            yield sse_event("chunk", {"text": latest_answer})

        log_tool_calls(session_id, collected_tool_calls)
        latest_answer = sanitize_ungrounded_kb_claim(latest_answer, collected_tool_calls)
        yield sse_event("done", {"answer": latest_answer})
    except Exception as exc:
        logger.exception("[CHAT_MM_STREAM_ERROR] session=%s error=%s", session_id, exc)
        yield sse_event("error", {"detail": str(exc)})
    finally:
        if latest_answer and not interrupted:
            save_message(session_id, "assistant", latest_answer)
            update_conversation(session_id)
            set_title(session_id, message or "（图片对话）")

def try_acquire_session(session_id: str) -> bool:
    sid = (session_id or "").strip()
    if not sid:
        return False
    lock = _SESSION_LOCKS.setdefault(sid, Lock())
    return lock.acquire(blocking=False)


def release_session(session_id: str) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    lock = _SESSION_LOCKS.get(sid)
    if lock is None:
        return
    if lock.locked():
        lock.release()

def extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts).strip()

    return str(content)

def extract_tool_calls_from_message(message) -> list[dict]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls and hasattr(message, "additional_kwargs"):
        tool_calls = message.additional_kwargs.get("tool_calls")
    if isinstance(tool_calls, list):
        return [call for call in tool_calls if isinstance(call, dict)]
    return []

def extract_tool_calls(result: dict) -> list[dict]:
    calls: list[dict] = []
    for message in result.get("messages", []):
        _extend_tool_calls_unique(calls, extract_tool_calls_from_message(message))
    return calls


def _tool_call_key(call: dict) -> tuple[str, str]:
    name = str(call.get("name", "")).strip()
    args = call.get("args", call.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    try:
        return name, json.dumps(args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return name, str(args)
        
def _should_skip_stream_node(node_name: str) -> bool:
    if node_name in _INTERNAL_STREAM_NODES:
        return True
    if _runtime == "langgraph-swarm" and node_name not in _FINAL_STREAM_NODES:
        return True
    return False

def _extend_tool_calls_unique(bucket: list[dict], new_calls: list[dict]) -> None:
    seen = {_tool_call_key(c) for c in bucket}
    for c in new_calls:
        k = _tool_call_key(c)
        if k in seen:
            continue
        seen.add(k)
        bucket.append(c)


def _current_turn_messages(messages: list[Any], current_user_message: str) -> list[Any]:
    if not isinstance(messages, list) or not messages:
        return []

    target = (current_user_message or "").strip()
    fallback_idx = None

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if getattr(msg, "type", "") != "human":
            continue
        fallback_idx = idx
        content = extract_text_from_content(getattr(msg, "content", "")).strip()
        if content == target or (target and content and (content in target or target in content)):
            return messages[idx + 1 :]

    if fallback_idx is not None:
        return messages[fallback_idx + 1 :]
    # LangGraph updates mode may emit only the node delta. Final swarm nodes
    # often return just [AIMessage(...)] instead of the full conversation state.
    if any(getattr(msg, "type", "") == "ai" for msg in messages):
        return messages
    return []

def extract_text_from_result(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "未获取到模型回复，请稍后重试。"

    for msg in reversed(messages):
        if getattr(msg, "type", "") != "ai":
            continue
        content = getattr(msg, "content", "")
        merged = extract_text_from_content(content).strip()
        return merged or str(content).strip()

    logger.warning(
        "[CHAT_NO_AI_MESSAGE] no ai message in result; message_types=%s",
        [getattr(m, "type", type(m).__name__) for m in messages[-8:]],
    )
    return "本轮未生成有效回答，请重试或换个问法。"


def _sanitize_echo_answer(answer: str, user_message: str) -> str:
    text = (answer or "").strip()
    user_text = (user_message or "").strip()
    if not text:
        return "本轮未生成有效回答，请重试或换个问法。"
    if user_text and text == user_text:
        logger.warning("[CHAT_ECHO_GUARD] detected echo answer, replace with fallback")
        return "本轮未生成有效回答，请重试或换个问法。"
    return text

_KB_GROUNDED_TOOLS = {
    "search_knowledge",
    "search_medical_kb",
    "search_knowledge_base",
}

def has_kb_tool_call(tool_calls: list[dict]) -> bool:
    for c in tool_calls:
        if str(c.get("name", "")).strip() in _KB_GROUNDED_TOOLS:
            return True
    return False

def sanitize_ungrounded_kb_claim(answer: str, tool_calls: list[dict]) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    
    if has_kb_tool_call(tool_calls):
        return text
    
    # 未调用 KB 工具，却出现“知识库引用”话术 => 改为非知识库建议
    if _KB_CLAIM_PAT.search(text):
        return (
            "本次回答未调用知识库检索工具，以下内容为通用建议，不能视为知识库结论。\n\n"
            + re.sub(r"参考文档：[\s\S]*$", "", text).strip()
        )
    
    return text

def _invoke_agent(session_id: str, message: str) -> tuple[str, list[dict]]:
    result = _agent.invoke(
        {"messages": [("user", message)]},
        config={"configurable": {"thread_id": session_id}},
    )
    answer = _sanitize_echo_answer(extract_text_from_result(result), message)
    tool_calls = extract_tool_calls(result)
    return answer, tool_calls

def invoke(session_id: str, message: str) -> tuple[str, list[dict]]:
    save_message(session_id, "user", message)
    answer, tool_calls = _invoke_agent(session_id, message)
    answer = sanitize_ungrounded_kb_claim(answer, tool_calls)
    save_message(session_id, "assistant", answer)
    update_conversation(session_id)
    set_title(session_id, message)
    return answer, tool_calls

def log_tool_calls(session_id: str, tool_calls: list[dict]) -> None:
    if tool_calls:
        for call in tool_calls:
            name = call.get("name", "unknown")
            args = call.get("args", call.get("arguments", {}))
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            logger.info("[TOOL_USED] session=%s name=%s args=%s", session_id, name, args)
    else:
        logger.info("[TOOL_SKIP] session=%s no tool call in this turn", session_id)

def sse_event(event: str, data: dict) -> str:

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def stream(session_id: str, message: str) -> Generator[str, None, None]:
    """SSE 流式生成器，controller 直接 yield from 即可。"""
    save_message(session_id, "user", message)
    latest_answer = ""
    interrupted = False
    try:
        if _runtime.startswith("langgraph") and _settings.agent_streaming:
            latest_answer, collected_tool_calls, interrupted = yield from _stream_langgraph_updates(
                session_id,
                {"messages": [("user", message)]},
                message,
            )
            if interrupted:
                return
            if not latest_answer:
                latest_answer, collected_tool_calls = _invoke_agent(session_id, message)
                latest_answer = _sanitize_echo_answer(latest_answer, message)
                yield sse_event("chunk", {"text": latest_answer})
        else:
            latest_answer, collected_tool_calls = _invoke_agent(session_id, message)
            latest_answer = _sanitize_echo_answer(latest_answer, message)
            yield sse_event("chunk", {"text": latest_answer})
        log_tool_calls(session_id, collected_tool_calls)
        latest_answer = sanitize_ungrounded_kb_claim(latest_answer, collected_tool_calls)
        yield sse_event("done", {"answer": latest_answer})
    except Exception as exc:
        logger.exception("[CHAT_STREAM_ERROR] session=%s error=%s", session_id, exc)
        yield sse_event("error", {"detail": str(exc)})
    finally:
        if latest_answer and not interrupted:
            save_message(session_id, "assistant", latest_answer)
            update_conversation(session_id)
            set_title(session_id, message)
            logger.info("[CHAT_RESULT] session=%s answer=%s", session_id, latest_answer)


def set_title(session_id: str, message: str) -> None:
    try:
        from db.messages import list_messages
        msgs = list_messages(session_id, limit=2)
        if len(msgs) <= 2:
            title = message[:30].strip()
            if title:
                update_title(session_id, title)
    except Exception:
        pass