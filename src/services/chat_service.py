import re
from config.logging_setup import setup_logging
from typing import Any, Generator
import logging
import json
from db.messages import save_message
from db.conversations import update_conversation, update_title

setup_logging()

_KB_CLAIM_PAT = re.compile(r"(根据知识库|知识库中|参考文档|来源：|source_dir|\.md)", re.IGNORECASE)
logger = logging.getLogger("chat.service")

# ---- 模块级单例，由 init() 注入 ----
_agent: Any = None
_settings: Any = None
_runtime: str = ""

def init(agent: Any, settings: Any, runtime: str) -> None:
    """由 app.py 启动时调一次，注入全局依赖。"""
    global _agent, _settings, _runtime
    _agent = agent
    _settings = settings
    _runtime = runtime

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
        calls.extend(extract_tool_calls_from_message(message))
    return calls

def extract_text_from_result(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response from model."

    last = messages[-1]
    content = getattr(last, "content", "")
    merged = extract_text_from_content(content)
    return merged or str(content)

_KB_GROUNDED_TOOLS = {"search_knowledge_base", "recommend_dishes"}

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
    answer = extract_text_from_result(result)
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
    sent_answer = ""
    collected_tool_calls: list[dict] = []
    try:
        if _runtime.startswith("langgraph") and _settings.agent_streaming:
            for update in _agent.stream(
                {"messages": [("user", message)]},
                config={"configurable": {"thread_id": session_id}},
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue
                for node_name, node_state in update.items():
                    if node_name == "supervisor":
                        continue
                    if not isinstance(node_state, dict):
                        continue
                    for msg in node_state.get("messages", []):
                        collected_tool_calls.extend(extract_tool_calls_from_message(msg))
                        if getattr(msg, "type", "") == "ai":
                            text = extract_text_from_content(getattr(msg, "content", "")).strip()
                            if not text:
                                continue
                            latest_answer = text
                            if text != sent_answer:
                                delta = text[len(sent_answer):] if text.startswith(sent_answer) else text
                                sent_answer = text
                                if delta:
                                    yield sse_event("chunk", {"text": delta})
            if not latest_answer:
                latest_answer, collected_tool_calls = _invoke_agent(session_id, message)
                yield sse_event("chunk", {"text": latest_answer})
        else:
            latest_answer, collected_tool_calls = _invoke_agent(session_id, message)
            yield sse_event("chunk", {"text": latest_answer})
        log_tool_calls(session_id, collected_tool_calls)
        latest_answer = sanitize_ungrounded_kb_claim(latest_answer, collected_tool_calls)
        yield sse_event("done", {"answer": latest_answer})
    except Exception as exc:
        logger.exception("[CHAT_STREAM_ERROR] session=%s error=%s", session_id, exc)
        yield sse_event("error", {"detail": str(exc)})
    finally:
        if latest_answer:
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