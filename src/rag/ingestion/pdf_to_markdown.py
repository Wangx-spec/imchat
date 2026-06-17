from __future__ import annotations

import re

from pathlib import Path
from llms.qwen_vl import QwenVLClient
from rag.ingestion.pdf_parser import ParsedPdfDoc

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _summary_block_for_image(
    image_name: str,
    image_path: str,
    *,
    vlm: QwenVLClient | None,
    static_url_prefix: str,
    assets_root: Path | None,
) -> str | None:
    if not vlm:
        return None
    
    cap = vlm.summarize_image(image_path=image_path)
    if not cap.ok or cap.image_type == "unsupported" or not cap.caption.strip():
        return None 
    image_id = _sanitize_image_id(image_name)
    image_url = _image_url_format(image_path, static_url_prefix=static_url_prefix, assets_root=assets_root)

    return "\n".join(
        _render_image_block(
            image_id=image_id,
            image_url=image_url,
            image_type=cap.image_type,
            is_medical=cap.is_medical,
            caption=cap.caption.strip(),
        )
    )

def _image_url_format(
    image_path: str,
    *,
    static_url_prefix: str,
    assets_root: Path | None,
) -> str:
    path = Path(image_path)

    if assets_root is not None:
        try:
            rel = path.resolve().relative_to(assets_root.resolve())
            return f"{static_url_prefix.rstrip('/')}/{rel.as_posix()}"
        except ValueError:
            pass
    
    return f"{static_url_prefix.rstrip('/')}/{path.name}"

def _render_image_block(
    *,
    image_id: str,
    image_url: str,
    image_type: str,
    is_medical: bool,
    caption: str,
) -> list[str]:
    return [
        f"### 图片：{image_id}",
        "",
        f"- 图片路径：{image_url}",
        f"- 类型：{image_type}",
        f"- is_medical：{str(is_medical).lower()}",
        f"- 摘要：{caption}",
        "",
        "> 注：该摘要由 Qwen-VL 生成，仅用于知识检索，不构成诊断结论。",
        "",
    ]


def render_markdown(
    parsed_doc: ParsedPdfDoc,
    topic: str,
    *,
    vlm: QwenVLClient | None = None,
    static_url_prefix: str = "/static/rag_assets/images",
    assets_root: Path | None = None,
) -> str:
    title = (parsed_doc.title or Path(parsed_doc.source_path).stem).strip()
    header: list[str] = [
        f"# {title}",
        "",
        f"> Source PDF: {parsed_doc.source_path.strip()}",
        f"> Topic: {topic}",
        "> Origin Type: pdf",
        "> Source Format: markdown_from_pdf",
        "",
    ]

    used: set[str] = set()

    def _replace(m: re.Match) -> str:
        name = m.group(1).strip()
        path = parsed_doc.images.get(name) or parsed_doc.images.get(Path(name).name)
        if not path:
            return m.group(0)
        
        used.add(name)
        block = _summary_block_for_image(
            name, path, vlm=vlm,
            static_url_prefix=static_url_prefix,
            assets_root=assets_root
        )
        return block if block else m.group(0)
    
    body = _MD_IMAGE_RE.sub(_replace, parsed_doc.markdown)

    # 兜底： Marker没在正文引用到的图片，统一追加到文末
    tail: list[str] = []
    for name, path in parsed_doc.images.items():
        if name in used:
            continue
            
        block = _summary_block_for_image(
            name, path, vlm=vlm,
            static_url_prefix=static_url_prefix,
            assets_root=assets_root
        )
        if block:
            tail.append(block)

    parts = ["\n".join(header), body.strip()]
    if tail:
        parts.append("## 未定位图片\n\n" + "\n\n".join(tail))
    return "\n\n".join(parts).strip() + "\n"

def _sanitize_image_id(name: str) -> str:
    stem = Path(name).name
    return re.sub(r"[^\w\-.]", "_", stem)