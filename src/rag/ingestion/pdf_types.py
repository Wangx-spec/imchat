from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedPdfDoc:
    title: str
    source_path: str
    markdown: str
    images: dict[str, str]
