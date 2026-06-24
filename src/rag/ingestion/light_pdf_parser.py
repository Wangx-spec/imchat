from __future__ import annotations

from pathlib import Path

from rag.ingestion.pdf_types import ParsedPdfDoc


class LightPdfTextTooShortError(ValueError):
    """Raised when a PDF has too little extractable text for light parsing."""


def parse_pdf_light(pdf_path: Path, *, min_chars: int = 200) -> ParsedPdfDoc:
    """Parse text-layer PDFs with PyMuPDF without downloading OCR/layout models."""
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("pymupdf_import_failed") from exc

    pdf_path = pdf_path.resolve()
    parts: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for idx, page in enumerate(doc, start=1):
            text = (page.get_text("text") or "").strip()
            if not text:
                continue
            parts.append(f"## 第 {idx} 页\n\n{text}")

    markdown = "\n\n".join(parts).strip()
    if len(markdown) < min_chars:
        raise LightPdfTextTooShortError("light_pdf_text_too_short")

    return ParsedPdfDoc(
        title=pdf_path.stem,
        source_path=str(pdf_path),
        markdown=markdown,
        images={},
    )
