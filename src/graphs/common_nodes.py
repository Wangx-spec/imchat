import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState

from agents.guardrails import LocalGuardrails
from agents.guardrails.image_guardrails import ImageGuardrailResult, check_image_attachments


logger = logging.getLogger(__name__)


def latest_user_message_text(state: MessagesState) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def latest_ai_message(state: MessagesState):
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "ai":
            return msg
    return None


def build_input_guardrail_node(guardrails: LocalGuardrails):
    def input_guardrail_node(state: MessagesState) -> dict:
        user_text = latest_user_message_text(state)
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


def build_image_input_guardrail_node(settings):
    def node(state: MessagesState) -> dict:
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


def build_image_caption_node(vlm):
    def node(state: MessagesState) -> dict:
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

        original_text = latest_user_message_text(state)
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


def build_output_guardrail_node(guardrails: LocalGuardrails):
    def output_guardrail_node(state: MessagesState) -> dict:
        user_text = latest_user_message_text(state)
        last_ai = latest_ai_message(state)
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
                    id=getattr(last_ai, "id", None),
                )
            ]
        }

    return output_guardrail_node
