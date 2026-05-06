from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ImageGuardrailResult:
    ok: bool
    user_message: str = ""
    reason: str = ""

def check_image_attachments(
    *,
    attachments: list[dict],
    allowed_mime: list[str],
    max_image_bytes: int,
    max_images: int,
) -> ImageGuardrailResult:
    if not attachments:
        return ImageGuardrailResult(ok=True)
    
    if len(attachments) > max_images:
        return ImageGuardrailResult(
            ok=False,
            reason="too_many_images",
            user_message=f"一次最多支持 {max_images} 张图片，请分多次上传。",
        )
    
    bad_mime: list[str] = []
    too_large: list[str] = []
    for att in attachments:
        mime = str(att.get("mime") or "").lower()
        size = int(att.get("size") or 0)
        if mime not in allowed_mime:
            bad_mime.append(att.get("image_id") or mime or "unknown")
        if size > max_image_bytes:
            too_large.append(att.get("image_id") or "unknown")

    if bad_mime:
        return ImageGuardrailResult(
            ok=False,
            reason="bad_mime",
            user_message=f"以下图片格式不支持：{bad_mime}。仅支持 jpg/png/webp。",
        )
    if too_large:
        return ImageGuardrailResult(
            ok=False,
            reason="too_large",
            user_message=f"以下图片超过 {max_image_bytes // 1024 // 1024}MB 限制：{too_large}。",
        )

    return ImageGuardrailResult(ok=True)