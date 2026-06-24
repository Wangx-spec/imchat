from __future__ import annotations
from pathlib import Path
import logging
from typing import Any

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from rag.ingestion.pdf_types import ParsedPdfDoc

logger = logging.getLogger(__name__)


def parse_pdf(
    pdf_path: Path, 
    image_dir: Path, 
    *, 
    min_image_bytes: int = 4096,
    model_dict: dict[str, Any] | None = None,
) -> ParsedPdfDoc:

    out_dir = (image_dir / pdf_path.stem).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = PdfConverter(artifact_dict=model_dict or create_model_dict())
    rendered = converter(str(pdf_path))
    markdown, _ext, images = text_from_rendered(rendered)  # images: {name: PIL.Image}

    saved: dict[str, str] = {}
    for name, pil_img in (images or {}).items():
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(target)
        if target.stat().st_size < min_image_bytes:
            target.unlink(missing_ok=True)
            continue
        saved[name] = str(target.resolve())

    return ParsedPdfDoc(
        title=pdf_path.stem,
        source_path=str(pdf_path.resolve()),
        markdown=markdown or "",
        images=saved,
    )
