from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_MEDICAL_TOPICS: set[str] = {
    "brain_tumor",
    "chest_xray",
    "skin_lesion",
    "diabetes",
}
_FILENAME_KEYWORD_MAP: list[tuple[str, str]] = [
    ("brain_tumor", "brain_tumor"),
    ("brain_tumors", "brain_tumor"),
    ("chest_xray", "chest_xray"),
    ("covid", "chest_xray"),
    ("skin_lesion", "skin_lesion"),
    ("diabetes", "diabetes"),
]


class MarkdownDataLoader:
    def scan_markdown_files(self, source_dirs: list[str]) -> list[Path]:
        files: list[Path] = []
        seen: set[str] = set()

        for d in source_dirs:
            root = Path(d)
            if not root.exists():
                logger.warning("RAG source dir not found: %s", d)
                continue

            for p in root.rglob("*.md"):
                # 去重
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    files.append(Path(key))

        # 排序
        files.sort(key=lambda x: str(x))
        return files

    def load_documents(self, source_dirs: list[str]) -> list[Document]:
        docs: list[Document] = []
        for path in self.scan_markdown_files(source_dirs):
            text = self._read_text(path)
            if not text.strip():
                continue

            metadata = self._build_parent_metadata(path, text)
            docs.append(Document(page_content=text, metadata=metadata))

        logger.info("Loaded %d parent documents", len(docs))
        return docs

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Skip unreadable markdown file: %s (%s)", path, exc)
            return ""

    def _extract_title(self, text: str, fallback: str) -> str:
        # 首个markdown一级标题
        m = re.search(r"(?m)^\s*#\s+(.+?)\s*$", text)
        if m:
            return m.group(1).strip()
        return fallback

    def _extract_header_fields(self, text: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for line in text.splitlines()[:20]:
            line = line.strip()
            if not line.startswith(">"):
                continue
            body = line[1:].strip()
            if ":" not in body:
                continue
            key, value = body.split(":", 1)
            fields[key.strip().lower()] = value.strip()
        return fields

    def _is_pdf_generated_markdown(self, path: Path) -> bool:
        return "parsed_md" in {p.lower() for p in path.parts}

    def _infer_category(self, path: Path) -> str:
        """
        推断文档主题。优先级：
        1. 路径里出现已知 topic 目录名（`source_dir/parsed_md/brain_tumor/xxx.md`）
        2. 文件名关键词命中
        3. 兜底 `general`
        """
        parts_lower = [p.lower() for p in path.parts]
        for p in parts_lower:
            if p in _MEDICAL_TOPICS:
                return p
        name_lower = path.stem.lower()
        for keyword, topic in _FILENAME_KEYWORD_MAP:
            if keyword in name_lower:
                return topic
        return "general"

    def _build_parent_metadata(self, path: Path, text: str) -> dict:
        headers = self._extract_header_fields(text)
        source_path = str(path.resolve())
        topic = headers.get("topic") or self._infer_category(path)
        is_pdf_md = self._is_pdf_generated_markdown(path)
        source_format = headers.get("source format") or (
            "markdown_from_pdf" if is_pdf_md else "markdown_authored"
        )
        origin_type = headers.get("origin type") or ("pdf" if is_pdf_md else "md")
        origin_file = headers.get("source pdf") or source_path

        return {
            "source": source_path,
            "title": self._extract_title(text, path.stem),
            "category": topic,
            "doc_type_topic": topic,
            "source_format": source_format,
            "origin_type": origin_type,
            "origin_file": origin_file,
            "parent_id": str(uuid.uuid4()),
            "doc_type": "parent",
        }
