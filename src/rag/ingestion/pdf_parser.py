from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import fitz  # pymupdf

logger = logging.getLogger(__name__)


@dataclass
class ParsedPdfDoc:
    title: str
    source_path: str
    raw_text: str
    pages: list[str]
    images: list[list[str]]


def _clean_page_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    return cleaned


def parse_pdf(pdf_path: Path, image_dir: Path, *, min_image_bytes: int = 4096) -> ParsedPdfDoc:

    # 图片处理
    image_dir = image_dir.resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    out_dir = image_dir / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 文本处理
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    images_per_page: list[list[str]] = []

    for page_idx, page in enumerate(doc, start=1):
        text = _clean_page_text(page.get_text("text"))
        if text:
            pages.append(text)

        page_imgs: list[str] = []
        for img_idx, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image", b"")
                ext = base_image.get("ext", "png")
                if len(image_bytes) < min_image_bytes:
                    continue
                
                target = out_dir / f"page_{page_idx}_fig_{img_idx}.{ext}"
                target.write_bytes(image_bytes)
                page_imgs.append(str(target.resolve()))
            except Exception as exc:
                logger.warning("extract_image_failed page=%d img=%d err=%s", 
                page_idx, img_idx, exc)
        images_per_page.append(page_imgs)

    raw_text = "\n\n".join([p for p in pages if p]).strip()
    return ParsedPdfDoc(
        title=pdf_path.stem,
        source_path=str(pdf_path.resolve()),
        raw_text=raw_text,
        pages=pages,
        images=images_per_page,
    )
