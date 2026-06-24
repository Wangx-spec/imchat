from __future__ import annotations

import json
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from actions.knowledge_base_tools import get_rag_service, set_rag_service
from config.settings import load_settings
from llms.openai_chat import build_openai_chat_model, build_router_chat_model
from rag.core.config import build_rag_config, sanitize_rag_config, validate_rag_config
from rag.core.service import RAGService
from rag.ingestion.light_pdf_parser import LightPdfTextTooShortError, parse_pdf_light
from rag.ingestion.pdf_to_markdown import render_markdown

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])
_REBUILD_LOCK = Lock()


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    include_answer: bool = False


class KnowledgeUploadRequest(BaseModel):
    rebuild: bool = False


@router.get("/stats")
def knowledge_stats() -> dict[str, Any]:
    service = get_rag_service()
    settings = load_settings()
    if service is None:
        return {
            "enabled": settings.rag_enabled,
            "ready": False,
            "reason": "service_not_initialized",
            "source_dirs": settings.rag_source_dirs,
            "index_dir": settings.rag_index_dir,
            "vector_db_provider": settings.vector_db_provider,
            "parent_count": 0,
            "child_count": 0,
        }
    stats = dict(service.stats())
    cfg = getattr(service, "cfg", None)
    stats.update(
        {
            "vector_db_provider": getattr(cfg, "vector_db_provider", settings.vector_db_provider),
            "rerank_enabled": getattr(cfg, "rerank_enabled", settings.rag_rerank_enabled),
            "parallel_recall": getattr(cfg, "parallel_recall", True),
            "cache_enabled": getattr(cfg, "cache_enabled", True),
        }
    )
    return stats


@router.get("/sources")
def knowledge_sources() -> dict[str, Any]:
    settings = load_settings()
    rows: list[dict[str, Any]] = []
    for raw_dir in settings.rag_source_dirs:
        root = Path(raw_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        if not root.exists():
            rows.append(
                {
                    "path": str(root),
                    "name": root.name,
                    "size": 0,
                    "modified_at": None,
                    "exists": False,
                    "source_dir": str(root),
                }
            )
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            st = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size": st.st_size,
                    "modified_at": st.st_mtime,
                    "exists": True,
                    "source_dir": str(root),
                }
            )
    return {"source_dirs": settings.rag_source_dirs, "items": rows}


@router.post("/search")
def knowledge_search(payload: KnowledgeSearchRequest) -> dict[str, Any]:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")
    service = get_rag_service()
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    if not bool(getattr(service, "ready", False)):
        raise HTTPException(status_code=503, detail="RAG service not ready")

    top_k = max(1, min(int(payload.top_k or 5), 20))
    start = time.monotonic()
    if payload.include_answer:
        answer_result = service.answer(query)
        ret = service.retrieve(query)
        answer = answer_result.answer
        sources = answer_result.sources or ret.sources
        debug = dict(answer_result.debug or ret.debug or {})
        insufficient_info = bool(answer_result.insufficient_info)
    else:
        ret = service.retrieve(query)
        answer = ""
        sources = ret.sources
        debug = dict(ret.debug or {})
        insufficient_info = bool(debug.get("insufficient_info", False))

    parents = [_doc_to_payload(doc) for doc in (ret.parents or [])[:top_k]]
    return {
        "ok": True,
        "query": query,
        "answer": answer,
        "sources": sources,
        "parents": parents,
        "debug": debug,
        "insufficient_info": insufficient_info,
        "elapsed_ms": round((time.monotonic() - start) * 1000, 1),
    }


@router.post("/upload")
async def knowledge_upload(
    file: UploadFile = File(...),
    rebuild: bool = False,
    pdf_parser: str = "light",
    topic: str = "uploaded",
) -> dict[str, Any]:
    target_root = _upload_root()
    filename = _safe_filename(file.filename or "document.md")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="file is empty")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large")

    saved: list[str] = []
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        text = content.decode("utf-8", errors="ignore")
        saved.extend(_save_json_docs(target_root, filename, text))
    elif suffix == ".md":
        text = content.decode("utf-8", errors="ignore")
        path = _safe_child_path(target_root, filename)
        path.write_text(text, encoding="utf-8")
        saved.append(str(path))
    elif suffix == ".txt":
        text = content.decode("utf-8", errors="ignore")
        path = _safe_child_path(target_root, f"{Path(filename).stem}.md")
        path.write_text(f"# {Path(filename).stem}\n\n{text}\n", encoding="utf-8")
        saved.append(str(path))
    elif suffix == ".pdf":
        saved.extend(_save_pdf_doc(target_root, filename, content, pdf_parser, topic))
    else:
        raise HTTPException(status_code=400, detail="only .md/.txt/.json/.pdf are supported")

    rebuilt = None
    if rebuild:
        rebuilt = knowledge_rebuild()
    return {"ok": True, "saved_files": saved, "rebuild": rebuilt}


@router.post("/rebuild")
def knowledge_rebuild() -> dict[str, Any]:
    if not _REBUILD_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="RAG rebuild already running")
    start = time.monotonic()
    try:
        settings = load_settings()
        cfg = sanitize_rag_config(build_rag_config(settings))
        validate_rag_config(cfg)
        service = get_rag_service()
        if service is None:
            service = RAGService(
                cfg,
                llm=build_openai_chat_model(settings),
                query_planner_llm=build_router_chat_model(settings),
            )
            set_rag_service(service)
        service.initialize(force_rebuild=True)
        stats = service.stats()
        return {
            "ok": True,
            "parent_count": stats.get("parent_count", 0),
            "child_count": stats.get("child_count", 0),
            "index_rebuilt": stats.get("index_rebuilt", False),
            "elapsed_ms": round((time.monotonic() - start) * 1000, 1),
        }
    finally:
        _REBUILD_LOCK.release()


def _doc_to_payload(doc: Any) -> dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    title = str(metadata.get("title") or metadata.get("source") or metadata.get("path") or "")
    source = str(metadata.get("source") or metadata.get("path") or metadata.get("file_path") or "")
    content = str(getattr(doc, "page_content", "") or "")
    return {
        "title": title,
        "source": source,
        "content": content[:1200],
        "metadata": metadata,
    }


def _upload_root() -> Path:
    settings = load_settings()
    if not settings.rag_source_dirs:
        raise HTTPException(status_code=400, detail="RAG_SOURCE_DIRS is empty")
    root = Path(settings.rag_source_dirs[0]).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = (root / "uploads").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_filename(name: str) -> str:
    base = Path(name).name.strip() or "document.md"
    base = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", base)
    return base[:120] or "document.md"


def _safe_child_path(root: Path, filename: str) -> Path:
    path = (root / _safe_filename(filename)).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise HTTPException(status_code=400, detail="invalid file path")
    return path


def _safe_topic(topic: str) -> str:
    value = (topic or "uploaded").strip().lower()
    value = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "_", value)
    return value[:80] or "uploaded"


def _save_pdf_doc(
    root: Path,
    filename: str,
    content: bytes,
    pdf_parser: str,
    topic: str,
) -> list[str]:
    parser = (pdf_parser or "light").strip().lower()
    if parser not in {"light", "marker"}:
        raise HTTPException(status_code=400, detail="unsupported_pdf_parser")

    pdf_dir = (root / "pdf").resolve()
    parsed_dir = (root / "parsed_md" / _safe_topic(topic)).resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = _safe_child_path(pdf_dir, filename)
    pdf_path.write_bytes(content)

    try:
        if parser == "light":
            parsed = parse_pdf_light(pdf_path)
            source_format = "markdown_from_pdf_light"
        else:
            try:
                from rag.ingestion.pdf_parser import parse_pdf
            except ImportError as exc:
                raise HTTPException(status_code=500, detail="marker-pdf 依赖未安装。") from exc
            settings = load_settings()
            parsed = parse_pdf(
                pdf_path=pdf_path,
                image_dir=Path(settings.multimodal_assets_dir),
                min_image_bytes=settings.multimodal_min_image_bytes,
            )
            source_format = "markdown_from_pdf"
    except LightPdfTextTooShortError as exc:
        raise HTTPException(
            status_code=400,
            detail="该 PDF 可能是扫描件或无文本层，请使用深度解析。",
        ) from exc
    except RuntimeError as exc:
        if str(exc) == "pymupdf_import_failed":
            raise HTTPException(status_code=500, detail="PyMuPDF 未安装，请安装 pymupdf。") from exc
        raise

    markdown = render_markdown(
        parsed_doc=parsed,
        topic=_safe_topic(topic),
        source_format=source_format,
    )
    md_path = _safe_child_path(parsed_dir, f"{Path(filename).stem}.md")
    md_path.write_text(markdown, encoding="utf-8")
    return [str(pdf_path), str(md_path)]


def _save_json_docs(root: Path, filename: str, text: str) -> list[str]:
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="json must be a list")
    stem = Path(filename).stem
    saved: list[str] = []
    for idx, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or f"{stem}-{idx}").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        safe = _safe_filename(f"{title or stem}-{idx}.md")
        path = _safe_child_path(root, safe)
        path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
        saved.append(str(path))
    if not saved:
        raise HTTPException(status_code=400, detail="json contains no valid documents")
    return saved
