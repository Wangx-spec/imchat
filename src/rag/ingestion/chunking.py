from __future__ import annotations

import re
import uuid
from typing import Tuple

from langchain_core.documents import Document

_IMAGE_BLOCK_RE = re.compile(
    r"^###\s+图片：(?P<image_id>[\w\-\.]+)\s*\n"
    r"(?:\s*\n)?"
    r"(?:- 图片路径：(?P<path>.+?)\s*\n)?"
    r"(?:- 类型：(?P<type>.+?)\s*\n)?"
    r"(?:- is_medical：(?P<medical>.+?)\s*\n)?"
    r"(?:- 摘要：(?P<caption>.+?)\s*\n)?",
    flags=re.MULTILINE,
)

def _extract_image_meta(chunk_text: str) -> list[dict]:
    out: list[dict] = []
    for m in _IMAGE_BLOCK_RE.finditer(chunk_text or ""):
        out.append({
            "image_id": m.group("image_id"),
            "image_path": (m.group("path") or "").strip(),
            "image_type": (m.group("type") or "general").strip(),
            "is_medical": (m.group("medical") or "false").strip().lower() == "true",
            "caption": (m.group("caption") or "").strip(),
        })
    return out


class ParentChildChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_parent_child(
        self, parents: list[Document]
    ) -> Tuple[list[Document], dict[str, Document], dict[str, str]]:
        children: list[Document] = []
        parent_map: dict[str, Document] = {}
        child_parent: dict[str, str] = {}

        for parent in parents:
            parent_id = parent.metadata["parent_id"]
            parent_map[parent_id] = parent

            sections = self._split_by_markdown_headers(parent.page_content)
            child_idx = 0
            for sec in sections:
                for part in self._split_long_text(sec):
                    text = part.strip()
                    if not text:
                        continue
                    child = self._make_child_doc(parent, text, child_idx)
                    children.append(child)

                    cid = child.metadata["child_id"]
                    child_parent[cid] = parent_id
                    child_idx += 1

        return children, parent_map, child_parent

    def _split_by_markdown_headers(self, text: str) -> list[str]:
        # 按 #/##/### 切分并保留标题
        lines = text.splitlines()
        blocks: list[list[str]] = []
        cur: list[str] = []

        header_re = re.compile(r"^\s*#{1,3}\s+")
        for line in lines:
            if header_re.match(line) and cur:
                blocks.append(cur)
                cur = [line]
            else:
                cur.append(line)

        if cur:
            blocks.append(cur)

        # 文档没有标题事，返回整篇
        if not blocks:
            return [text]

        return ["\n".join(b).strip() for b in blocks if "\n".join(b).strip()]

    def _split_long_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        step = max(1, self.chunk_size - self.chunk_overlap)
        out: list[str] = []
        i = 0
        n = len(text)

        while i < n:
            j = min(i + self.chunk_size, n)
            out.append(text[i:j])
            if j >= n:
                break
            i += step

        return out

    def _make_child_doc(self, parent: Document, chunk_text: str, idx: int) -> Document:
        child_id = str(uuid.uuid4())
        meta = dict(parent.metadata)
        meta["images"] = _extract_image_meta(chunk_text)
        meta["has_image"] = bool(meta["images"])
        meta.update(
            {
                "child_id": child_id,
                "parent_id": parent.metadata["parent_id"],
                "chunk_index": idx,
                "doc_type": "child",
                "chunk_size": len(chunk_text),
            }
        )
        return Document(page_content=chunk_text, metadata=meta)
