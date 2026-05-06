from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """\
你是一个医学知识库的图片摘要器。
只描述图片中可见内容，不输出诊断结论，不判断良恶性，不替代医生阅片。
返回 JSON：
{
  "caption": "<200字以内的中文描述>",
  "image_type": "medical_imaging | medical_diagram | chart | general | unsupported",
  "is_medical": true,
  "is_diagnostic_request": false,
  "uncertain_points": []
}
只返回 JSON，不要输出其他文字。
"""

@dataclass(frozen=True)
class QwenVLConfig:
    model: str
    api_key: str
    base_url: str
    timeout_ms: int = 10000


@dataclass
class ImageCaption:
    ok: bool
    caption: str
    image_type: str
    is_medical: bool
    is_diagnostic_request: bool
    uncertain_points: list[str]
    error: str | None = None

class QwenVLClient:
    def __init__(self, cfg: QwenVLConfig) -> None:
        self.cfg = cfg
        self._endpoint = f"{cfg.base_url.rstrip('/')}/chat/completions"
    
    def summarize_image(
        self,
        *,
        image_path: str | None = None,
        image_url: str | None = None,
        user_question: str = "请生成图片摘要。",
    ) -> ImageCaption:
        if not image_path and not image_url:
            return self._failed("no_image_input")
        
        try:
            image_payload = image_url or _local_image_to_data_url(image_path)
            payload = {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_payload}},
                            {"type": "text", "text": user_question},
                        ],
                    }
                ],
                "temperature": 0,
            }

            headers = {
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            }

            with httpx.Client(timeout=self.cfg.timeout_ms / 1000) as client:
                resp = client.post(self._endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            parsed = _parse_json_lenient(text)

            return ImageCaption(
                ok=True,
                caption=str(parsed.get("caption", "")).strip(),
                image_type=str(parsed.get("image_type", "general")).strip(),
                is_medical=bool(parsed.get("is_medical", False)),
                is_diagnostic_request=bool(parsed.get("is_diagnostic_request", False)),
                uncertain_points=[
                    str(x) for x in parsed.get("uncertain_points", []) if x
                ],
            )
        except Exception as exc:
            logger.exception("[QWEN_VL] summarize_image failed: %s", exc)
            return self._failed(str(exc))
        
    def _failed(self, error: str) -> ImageCaption:
        return ImageCaption(
            ok=False,
            caption="",
            image_type="unsupported",
            is_medical=False,
            is_diagnostic_request=False,
            uncertain_points=[],
            error=error,
        )
    
def build_qwen_vl_client(settings) -> QwenVLClient | None:
    if not getattr(settings, "multimodal_enabled", False):
        return None

    api_key = (getattr(settings, "multimodal_api_key", "") or "").strip()
    if not api_key:
        logger.warning("[QWEN_VL] missing multimodal api key")
        return None

    return QwenVLClient(
        QwenVLConfig(
            model=settings.multimodal_model,
            api_key=api_key,
            base_url=settings.multimodal_base_url,
            timeout_ms=settings.multimodal_timeout_ms,
        )
    )

def _local_image_to_data_url(path: str) -> str:
    p = Path(path)
    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"

def _parse_json_lenient(text: str) -> dict:
    raw = (text or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]+\}", raw)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}