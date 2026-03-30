import json
import logging
from pathlib import Path
import sys
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Support running via `python web/app.py` from `src/`.
if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from agents.dialog_agent import build_dialog_agent
from config.logging_setup import setup_logging
from config.settings import load_settings
from memory.session_memory import ChatSessionMemory


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


class ResetRequest(BaseModel):
    session_id: str


logger = logging.getLogger("chat.web")


def _extract_tool_calls(result: dict) -> list[dict]:
    calls: list[dict] = []
    for message in result.get("messages", []):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and hasattr(message, "additional_kwargs"):
            tool_calls = message.additional_kwargs.get("tool_calls")
        if tool_calls:
            for call in tool_calls:
                if isinstance(call, dict):
                    calls.append(call)
    return calls


def _extract_text_from_result(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response from model."

    last = messages[-1]
    content = getattr(last, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        merged = "".join(text_parts).strip()
        return merged or str(content)

    return str(content)


setup_logging()
app = FastAPI(title="LangChain Chat UI")
_settings = load_settings()
_agent = build_dialog_agent(_settings)
_memory_map: dict[str, ChatSessionMemory] = {}
_memory_lock = Lock()
_index_file = Path(__file__).resolve().parent / "static" / "index.html"


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

    try:
        result = _agent.invoke({"messages": history})
        tool_calls = _extract_tool_calls(result)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name", "unknown")
                args = call.get("args", call.get("arguments", {}))
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                logger.info(
                    "[TOOL_USED] session=%s name=%s args=%s", session_id, name, args
                )
        else:
            logger.info("[TOOL_SKIP] session=%s no tool call in this turn", session_id)
        answer = _extract_text_from_result(result)
    except Exception as exc:
        logger.exception("[CHAT_ERROR] session=%s error=%s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    with _memory_lock:
        memory.append_assistant(answer)
    logger.info("[CHAT_RESULT] session=%s answer=%s", session_id, answer)

    return ChatResponse(answer=answer)


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
