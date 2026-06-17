from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


class IndexStoreProtocol(Protocol):
    vectorstore: Any | None
    last_meta_match: bool | None
    last_meta_reason: str | None

    def exists(self) -> bool: ...
    def build(self, children: list[Document]) -> None: ...
    def save(self) -> None: ...
    def load(self, expected_children_count: int | None = None) -> bool: ...
    def as_retriever(self, k: int) -> Any: ...


def _build_embeddings(cfg) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=cfg.embedding_api_key,
        model=cfg.embedding_model,
        base_url=cfg.embedding_base_url,
        dimensions=cfg.embedding_dimensions,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )


class LocalFAISSIndexStore:
    """Manage local FAISS index lifecycle for RAG child chunks."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.index_dir = Path(cfg.index_dir)
        self.embeddings = _build_embeddings(cfg)
        self.vectorstore: FAISS | None = None
        self._last_children_count = -1
        self.last_meta_match: bool | None = None
        self.last_meta_reason: str | None = None

    def exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def build(self, children: list[Document]) -> None:
        if not children:
            raise ValueError("children is empty, cannot build vector index")
        self._last_children_count = len(children)
        self.vectorstore = FAISS.from_documents(children, self.embeddings)
        logger.info("FAISS index built with %d child documents", len(children))

    def save(self) -> None:
        if self.vectorstore is None:
            raise ValueError("vectorstore is None, build or load before save")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))
        meta: dict[str, Any] = self._build_signature(self._last_children_count)
        self._save_meta(meta)
        logger.info("FAISS index saved to %s", self.index_dir)

    def load(self, expected_children_count: int | None = None) -> bool:
        if not self.exists():
            self.last_meta_match = None
            self.last_meta_reason = "index_files_missing"
            return False
        if expected_children_count is None:
            self.last_meta_match = False
            self.last_meta_reason = "expected_children_count_missing"
            return False
        expected = self._build_signature(expected_children_count)
        if not self._is_meta_match(expected):
            return False
        self.vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return True

    def as_retriever(self, k: int) -> Any:
        if self.vectorstore is None:
            raise ValueError("vectorstore is None, build or load before save")
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / "index.meta.json"

    def _build_signature(self, children_count: int) -> dict[str, Any]:
        return {
            "source_dirs": sorted(self.cfg.source_dirs),
            "embedding_model": self.cfg.embedding_model,
            "embedding_dimensions": int(self.cfg.embedding_dimensions),
            "chunk_size": int(self.cfg.chunk_size),
            "chunk_overlap": int(self.cfg.chunk_overlap),
            "children_count": int(children_count),
        }

    def _save_meta(self, meta: dict[str, Any]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_meta(self) -> dict | None:
        if not self._meta_path.exists():
            return None
        try:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _is_meta_match(self, expected_meta: dict) -> bool:
        stored = self._load_meta()
        if stored is None:
            self.last_meta_match = False
            self.last_meta_reason = "meta_missing_or_corrupted"
            return False
        if stored != expected_meta:
            self.last_meta_match = False
            self.last_meta_reason = "meta_mismatch"
            return False
        self.last_meta_match = True
        self.last_meta_reason = "meta_match"
        return True


class QdrantIndexStore:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.embeddings = _build_embeddings(cfg)
        self.vectorstore: QdrantVectorStore | None = None
        self.last_meta_match: bool | None = None
        self.last_meta_reason: str | None = None
        self.client = QdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
        )
        self.collection = cfg.qdrant_collection

    def exists(self) -> bool:
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:
            return False

    def build(self, children: list[Document]) -> None:
        if not children:
            raise ValueError("children is empty, cannot build vector index")
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=int(self.cfg.embedding_dimensions),
                distance=qmodels.Distance.COSINE,
            ),
        )
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection,
            embedding=self.embeddings,
        )
        self.vectorstore.add_documents(children)
        self.last_meta_match = True
        self.last_meta_reason = "collection_rebuilt"
        logger.info("Qdrant collection rebuilt with %d child documents", len(children))

    def save(self) -> None:
        # Qdrant 数据已持久化在服务端
        return None

    def load(self, expected_children_count: int | None = None) -> bool:
        if not self.exists():
            self.last_meta_match = None
            self.last_meta_reason = "collection_missing"
            return False
        try:
            info = self.client.get_collection(self.collection)
            points_count = int(getattr(info, "points_count", 0) or 0)
            if expected_children_count is not None and points_count != int(expected_children_count):
                self.last_meta_match = False
                self.last_meta_reason = "points_count_mismatch"
                return False
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection,
                embedding=self.embeddings,
            )
            self.last_meta_match = True
            self.last_meta_reason = "collection_loaded"
            return True
        except Exception:
            self.last_meta_match = False
            self.last_meta_reason = "load_failed"
            return False

    def as_retriever(self, k: int) -> Any:
        if self.vectorstore is None:
            raise ValueError("vectorstore is None, build or load before save")
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def build_index_store(cfg) -> IndexStoreProtocol:
    provider = str(getattr(cfg, "vector_db_provider", "faiss") or "faiss").strip().lower()
    if provider == "qdrant":
        return QdrantIndexStore(cfg)
    return LocalFAISSIndexStore(cfg)
