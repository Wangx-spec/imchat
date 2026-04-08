import json
import logging
from pathlib import Path
import sys
from threading import Lock
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# Support running via `python web/app.py` from `src/`.
if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
from agents.dialog_agent import build_dialog_runtime
from config.logging_setup import setup_logging
from config.settings import load_settings
from memory.session_memory import ChatSessionMemory
from rag.bootstrap import bootstrap_rag
import re
from rag.forced_route import should_force_kb, call_kb, build_forced_kb_answer

_KB_CLAIM_PAT = re.compile(r"(根据知识库|知识库中|参考文档|来源：|source_dir|\.md)", re.IGNORECASE)

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


class ResetRequest(BaseModel):
    session_id: str


logger = logging.getLogger("chat.web")


def _extract_text_from_content(content) -> str:
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


def _extract_tool_calls_from_message(message) -> list[dict]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls and hasattr(message, "additional_kwargs"):
        tool_calls = message.additional_kwargs.get("tool_calls")
    if isinstance(tool_calls, list):
        return [call for call in tool_calls if isinstance(call, dict)]
    return []


def _extract_tool_calls(result: dict) -> list[dict]:
    calls: list[dict] = []
    for message in result.get("messages", []):
        calls.extend(_extract_tool_calls_from_message(message))
    return calls


def _extract_text_from_result(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response from model."

    last = messages[-1]
    content = getattr(last, "content", "")
    merged = _extract_text_from_content(content)
    return merged or str(content)


def _log_tool_calls(session_id: str, tool_calls: list[dict]) -> None:
    if tool_calls:
        for call in tool_calls:
            name = call.get("name", "unknown")
            args = call.get("args", call.get("arguments", {}))
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            logger.info("[TOOL_USED] session=%s name=%s args=%s", session_id, name, args)
    else:
        logger.info("[TOOL_SKIP] session=%s no tool call in this turn", session_id)


def _invoke_chat(history) -> tuple[str, list[dict]]:
    result = _agent.invoke({"messages": history})
    return _extract_text_from_result(result), _extract_tool_calls(result)


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def _has_kb_tool_call(tool_calls: list[dict]) -> bool:
    for c in tool_calls:
        if str(c.get("name", "")).strip() == "search_knowledge_base":
            return True
    return False

def _sanitize_ungrounded_kb_claim(answer: str, tool_calls: list[dict]) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    
    if _has_kb_tool_call(tool_calls):
        return text
    
    # 未调用 KB 工具，却出现“知识库引用”话术 => 改为非知识库建议
    if _KB_CLAIM_PAT.search(text):
        return (
            "本次回答未调用知识库检索工具，以下内容为通用建议，不能视为知识库结论。\n\n"
            + re.sub(r"参考文档：[\s\S]*$", "", text).strip()
        )
    
    return text



setup_logging()
app = FastAPI(title="Chat UI")
_settings = load_settings()
_agent, _runtime = build_dialog_runtime(_settings)
_memory_map: dict[str, ChatSessionMemory] = {}
_memory_lock = Lock()
_index_file = Path(__file__).resolve().parent / "static" / "index.html"
# module globals
_rag_ok = False
_rag_reason = "not_bootstrapped"

# try:
#     _rag_ok, _rag_reason = bootstrap_rag(_settings)
#     logger.info("[RAG_BOOTSTRAP] ok=%s reason=%s", _rag_ok, _rag_reason)
# except Exception as exc:
#     _rag_ok, _rag_reason = False, str(exc)
#     logger.warning("[RAG_BOOTSTRAP] failed=%s", exc)

@app.on_event("startup")
def on_startup() -> None:
    global _rag_ok, _rag_reason
    if not _settings.rag_enabled:
        _rag_ok, _rag_reason = False, "disabled"
        logger.info("rag_bootstrap_skipped reason=disabled")
        return
    try:
        _rag_ok, _rag_reason = bootstrap_rag(_settings)
        if _rag_ok:
            logger.info("rag_bootstrap_ok reason=%s", _rag_reason)
        else:
            logger.warning("rag_bootstrap_failed reason=%s", _rag_reason)
    except Exception as exc:
        _rag_ok, _rag_reason = False, f"init_error:{exc}"
        logger.warning("rag_bootstrap_failed reason=%s", exc)

@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "rag_ready": _rag_ok,
        "rag_reason": _rag_reason,
        "runtime": _runtime,
    }

@app.get("/")
def index() -> FileResponse:
    return FileResponse(_index_file)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    session_id = payload.session_id.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    with _memory_lock:
        memory = _memory_map.setdefault(session_id, ChatSessionMemory())
        memory.append_user(message)
        history = memory.messages()
    logger.info("[CHAT_REQUEST] session=%s message=%s", session_id, message)

    if _settings.rag_force_tool_route and should_force_kb(message):
        kb = call_kb(message)
        answer = build_forced_kb_answer(message, kb, _settings)
        with _memory_lock:
            memory.append_assistant(answer)
        logger.info(
            "[FORCED_KB] session=%s forced=True ok=%s error=%s",
            session_id, kb.get("ok"), kb.get("error")
        )
        logger.info("[CHAT_RESULT] session=%s answer=%s", session_id, answer)
        return ChatResponse(answer=answer)

    try:
        answer, tool_calls = _invoke_chat(history)
        _log_tool_calls(session_id, tool_calls)
        answer = _sanitize_ungrounded_kb_claim(answer, tool_calls)
    except Exception as exc:
        logger.exception("[CHAT_ERROR] session=%s error=%s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    with _memory_lock:
        memory.append_assistant(answer)
    logger.info("[CHAT_RESULT] session=%s answer=%s", session_id, answer)

    return ChatResponse(answer=answer)


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    message = payload.message.strip()
    session_id = payload.session_id.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    with _memory_lock:
        memory = _memory_map.setdefault(session_id, ChatSessionMemory())
        memory.append_user(message)
        history = memory.messages()
        
    logger.info("[CHAT_STREAM_REQUEST] session=%s message=%s", session_id, message)

    def event_generator():
        latest_answer = ""
        sent_answer = ""
        collected_tool_calls: list[dict] = []

        try:
            if _settings.rag_force_tool_route and should_force_kb(message):
                kb = call_kb(message)
                forced_answer = build_forced_kb_answer(message, kb, _settings)
                logger.info(
                    "[FORCED_KB] session=%s forced=True ok=%s error=%s",
                    session_id, kb.get("ok"), kb.get("error")
                )
                yield _sse_event("chunk", {"text": forced_answer})
                yield _sse_event("done", {"answer": forced_answer})
                with _memory_lock:
                    memory.append_assistant(forced_answer)
                logger.info("[CHAT_RESULT] session=%s answer=%s", session_id, forced_answer)
                return
            if _runtime == "langgraph" and _settings.agent_streaming:
                for update in _agent.stream({"messages": history}, stream_mode="updates"):
                    if not isinstance(update, dict):
                        continue
                    for node_state in update.values():
                        if not isinstance(node_state, dict):
                            continue
                        messages = node_state.get("messages", [])
                        for msg in messages:
                            collected_tool_calls.extend(_extract_tool_calls_from_message(msg))
                            if getattr(msg, "type", "") == "ai":
                                text = _extract_text_from_content(getattr(msg, "content", "")).strip()
                                if not text:
                                    continue
                                latest_answer = text
                                if text != sent_answer:
                                    delta = text
                                    if text.startswith(sent_answer):
                                        delta = text[len(sent_answer):]
                                    sent_answer = text
                                    if delta:
                                        yield _sse_event("chunk", {"text": delta})

                if not latest_answer:
                    latest_answer, collected_tool_calls = _invoke_chat(history)
                    yield _sse_event("chunk", {"text": latest_answer})
            else:
                latest_answer, collected_tool_calls = _invoke_chat(history)
                yield _sse_event("chunk", {"text": latest_answer})

            _log_tool_calls(session_id, collected_tool_calls)
            latest_answer = _sanitize_ungrounded_kb_claim(latest_answer, collected_tool_calls)
            yield _sse_event("done", {"answer": latest_answer})
        except Exception as exc:
            logger.exception("[CHAT_STREAM_ERROR] session=%s error=%s", session_id, exc)
            yield _sse_event("error", {"detail": str(exc)})
        finally:
            if latest_answer:
                with _memory_lock:
                    memory.append_assistant(latest_answer)
                logger.info("[CHAT_RESULT] session=%s answer=%s", session_id, latest_answer)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/reset")
def reset(payload: ResetRequest) -> dict:
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    with _memory_lock:
        _memory_map.pop(session_id, None)
    logger.info("[CHAT_RESET] session=%s", session_id)

    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
