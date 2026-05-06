from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from config.settings import load_settings
from services.chat_service import (
    invoke,
    stream,
    stream_multimodal,
    log_tool_calls,
    sanitize_ungrounded_kb_claim,
    try_acquire_session,
    release_session,
)
from db.messages import list_messages
import logging

logger = logging.getLogger("chat.web")

from db.conversations import (
    create_conversation,
    list_conversations,
    update_conversation,
)

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

    if not try_acquire_session(session_id):
        raise HTTPException(status_code=409, detail="This conversation is already processing another request.")

    try:
        answer, tool_calls = invoke(session_id, message)
        log_tool_calls(session_id, tool_calls)
        # answer = sanitize_ungrounded_kb_claim(answer, tool_calls)
    except Exception as exc:
        logger.exception("[CHAT_ERROR] session=%s error=%s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
    finally:
        release_session(session_id)

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

    if not try_acquire_session(session_id):
        raise HTTPException(status_code=409, detail="This conversation is already processing another request.")

    logger.info("[CHAT_STREAM_REQUEST] session=%s message=%s", session_id, message)

    def guarded_stream():
        try:
            yield from stream(session_id, message)
        finally:
            release_session(session_id)

    return StreamingResponse(
        guarded_stream(),
        media_type="text/event-stream",
    )


@router.post("/chat/multimodal/stream")
async def chat_multimodal_stream(
    session_id: str = Form(...),
    message: str = Form(""),
    images: list[UploadFile] = File(default_factory=list),
) -> StreamingResponse:
    sid = session_id.strip()
    text = message.strip()

    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not text and not images:
        raise HTTPException(status_code=400, detail="message or image is required")

    if not try_acquire_session(sid):
        raise HTTPException(status_code=409, detail="This conversation is already processing another request.")

    try:
        settings = load_settings()
        if images and not settings.multimodal_enabled:
            raise HTTPException(status_code=400, detail="multimodal disabled by config")

        attachments = await _persist_uploaded_images(sid, images, settings)
    except HTTPException:
        release_session(sid)
        raise
    except Exception as exc:
        release_session(sid)
        logger.exception("[CHAT_MM_UPLOAD_ERROR] session=%s error=%s", sid, exc)
        raise HTTPException(status_code=500, detail=f"Upload error: {exc}") from exc

    logger.info(
        "[CHAT_MM_STREAM_REQUEST] session=%s image_count=%d message=%s",
        sid,
        len(attachments),
        text,
    )

    def guarded_stream():
        try:
            yield from stream_multimodal(sid, text, attachments)
        finally:
            release_session(sid)

    return StreamingResponse(
        guarded_stream(),
        media_type="text/event-stream",
    )


async def _persist_uploaded_images(session_id: str, files: list[UploadFile], settings) -> list[dict]:
    if not files:
        return []

    if len(files) > settings.multimodal_max_images_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"一次最多支持 {settings.multimodal_max_images_per_request} 张图片，请分多次上传。",
        )

    base_dir = Path(settings.multimodal_uploads_dir) / session_id
    base_dir.mkdir(parents=True, exist_ok=True)

    attachments: list[dict] = []
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for upload in files:
        original_name = upload.filename or ""
        ext = Path(original_name).suffix.lower()
        if ext not in allowed_exts:
            ext = ".bin"

        image_id = uuid.uuid4().hex
        target = base_dir / f"{image_id}{ext}"
        with target.open("wb") as out_f:
            shutil.copyfileobj(upload.file, out_f)

        size = target.stat().st_size
        public_url = f"{settings.multimodal_uploads_url_prefix.rstrip('/')}/{session_id}/{target.name}"
        attachments.append(
            {
                "image_id": image_id,
                "image_path": str(target.resolve()),
                "public_url": public_url,
                "mime": upload.content_type or "",
                "size": size,
                "filename": original_name,
            }
        )

    return attachments


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