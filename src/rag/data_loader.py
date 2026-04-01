from __future__ import annotations
import logging
from pathlib import Path
import re
import uuid
from langchain_core.documents import Document
logger = logging.getLogger(__name__)


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

    def _infer_category(self, path: Path) -> str:
        s = str(path).replace("\\", "/")
        if "/HowToCook/" in s:
            parts = path.parts
            # 尝试拿 HowToCook 下一级目录作为分类
            try:
                idx = parts.index("HowToCook")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
            except ValueError:
                pass
            return "howtocook"
        return "general"

    def _build_parent_metadata(self, path: Path, text: str) -> dict:
        return {
            "source": str(path.resolve()),
            "title": self._extract_title(text, path.stem),
            "category": self._infer_category(path),
            "parent_id": str(uuid.uuid4()),
            "doc_type": "parent",
        }
