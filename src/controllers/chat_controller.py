from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.chat_service import (
    invoke,
    stream,
    log_tool_calls,
    sanitize_ungrounded_kb_claim,
)
import logging

logger = logging.getLogger("chat.web")

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

class ResetRequest(BaseModel):
    session_id: str



@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    session_id = payload.session_id.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    try:
        answer, tool_calls = invoke(session_id, message)
        log_tool_calls(session_id, tool_calls)
        answer = sanitize_ungrounded_kb_claim(answer, tool_calls)
    except Exception as exc:
        logger.exception("[CHAT_ERROR] session=%s error=%s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    logger.info("[CHAT_RESULT] session=%s answer=%s", session_id, answer)
    return ChatResponse(answer=answer)


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    message = payload.message.strip()
    session_id = payload.session_id.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    logger.info("[CHAT_STREAM_REQUEST] session=%s message=%s", session_id, message)
    return StreamingResponse(
        stream(session_id, message),
        media_type="text/event-stream",
    )

@router.post("/reset")
def reset(payload: ResetRequest) -> dict:
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    logger.info(
        "[CHAT_RESET] session=%s note=langgraph_checkpointer_in_use;reset_by_new_session_id",
        session_id,
    )
    return {"ok": True}