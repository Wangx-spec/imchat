from __future__ import annotations

from pathlib import Path
from llms.qwen_vl import QwenVLClient
from rag.ingestion.pdf_parser import ParsedPdfDoc


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()

def _image_id_format(image_path: str, page_idx: int, image_idx: int) -> str:
    stem = Path(image_path).stem
    return f"{stem}_p{page_idx}_f{image_idx}"

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

def _append_page_images(
    parts: list[str],
    *,
    page_idx: int,
    image_paths: list[str],
    vlm: QwenVLClient | None,
    static_url_prefix: str,
    assets_root: Path | None,
) -> None:
    if not image_paths or vlm is None:
        return
    for image_idx, image_path in enumerate(image_paths, start=1):
        caption_obj = vlm.summarize_image(image_path=image_path)
        if not caption_obj.ok:
            continue
        if caption_obj.image_type == "unsupported":
            continue
        if not caption_obj.caption.strip():
            continue
        image_id = _image_id_format(image_path, page_idx, image_idx)
        image_url = _image_url_format(
            image_path,
            static_url_prefix=static_url_prefix,
            assets_root=assets_root,
        )
        parts.extend(
            _render_image_block(
                image_id=image_id,
                image_url=image_url,
                image_type=caption_obj.image_type,
                is_medical=caption_obj.is_medical,
                caption=caption_obj.caption.strip(),
            )
        )

def render_markdown(
    parsed_doc: ParsedPdfDoc,
    topic: str,
    *,
    vlm: QwenVLClient | None = None,
    static_url_prefix: str = "/static/rag_assets/images",
    assets_root: Path | None = None,
) -> str:
    title = (parsed_doc.title or Path(parsed_doc.source_path).stem).strip()
    source_pdf = parsed_doc.source_path.strip()
    parts: list[str] = [
        f"# {title}",
        "",
        f"> Source PDF: {source_pdf}",
        f"> Topic: {topic}",
        "> Origin Type: pdf",
        "> Source Format: markdown_from_pdf",
        "",
    ]
    page_count = max(len(parsed_doc.pages), len(parsed_doc.images or []))
    if page_count > 0:
        for idx in range(page_count):
            page_no = idx + 1
            page_text = ""
            if idx < len(parsed_doc.pages):
                page_text = _normalize_text(parsed_doc.pages[idx])
            image_paths: list[str] = []
            if idx < len(parsed_doc.images):
                image_paths = parsed_doc.images[idx] or []
            parts.append(f"## Page {page_no}")
            parts.append("")
            if page_text:
                parts.append(page_text)
                parts.append("")
            _append_page_images(
                parts,
                page_idx=page_no,
                image_paths=image_paths,
                vlm=vlm,
                static_url_prefix=static_url_prefix,
                assets_root=assets_root,
            )
            parts.append("---")
            parts.append("")
    else:
        raw_text = _normalize_text(parsed_doc.raw_text)
        parts.append("## Content")
        parts.append("")
        parts.append(raw_text if raw_text else "_No extractable text._")
        parts.append("")
    return "\n".join(parts).strip() + "\n"
