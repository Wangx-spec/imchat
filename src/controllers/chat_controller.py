from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.chat_service import (
    invoke,
    stream,
    log_tool_calls,
)
from db.messages import list_messages
from db.conversations import delete_conversation
import logging

logger = logging.getLogger("chat.web")

from db.conversations import (
    create_conversation,
    list_conversations,
    update_conversation,
)
import uuid

router = APIRouter(prefix="/api", tags=["chat"])

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

class ResetRequest(BaseModel):
    session_id: str

class NewConversationResponse(BaseModel):
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
        # answer = sanitize_ungrounded_kb_claim(answer, tool_calls)
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
    new_id = str(uuid.uuid4())
    create_conversation(new_id)
    logger.info("[CHAT_RESET] old=%s new=%s", session_id, new_id)
    return {"ok": True, "new_session_id": new_id}

@router.post("/conversations", response_model=NewConversationResponse)
def new_conversation() -> NewConversationResponse:
    session_id = str(uuid.uuid4())
    create_conversation(session_id)
    return NewConversationResponse(session_id=session_id)

@router.get("/conversations")
def get_conversations() -> list[dict]:
    return list_conversations()

@router.get("/conversations/{session_id}/messages")
def get_messages(session_id: str) -> list[dict]:
    return list_messages(session_id)

@router.delete("/conversations/{session_id}")
def remove_conversation(session_id: str) -> dict:
    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    ok = delete_conversation(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="conversation not found")

    return {"ok": True}