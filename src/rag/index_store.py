from __future__ import annotations
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import Any
import logging
import json
logger = logging.getLogger(__name__)

class LocalFAISSIndexStore:
    """Manage local FAISS index lifecycle for RAG child chunks."""

    def __init__(self, cfg) -> None:
        """Initialize embedding client and index storage path from RAG config."""
        self.cfg = cfg
        self.index_dir = Path(cfg.index_dir)
        self.embeddings = OpenAIEmbeddings(
            api_key=cfg.embedding_api_key,
            model=cfg.embedding_model,
            base_url=cfg.embedding_base_url,
            dimensions=cfg.embedding_dimensions,
            check_embedding_ctx_length=False,
            chunk_size=10  # 关键：每批最多10条数据，避免内存溢出
        )
        self.vectorstore: FAISS | None = None
        self._last_children_count = -1
        self.last_meta_match: bool | None = None
        self.last_meta_reason: str | None = None

    def exists(self) -> bool:
        """Check whether a previously saved FAISS index exists on disk."""
        # FAISS.save_local 会生成 index.faiss 和 index.pkl
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def build(self, children: list[Document]) -> None:
        """Build a FAISS vector index from child documents in memory."""
        if not children:
            raise ValueError("children is empty, cannot build vector index")

        self._last_children_count = len(children)  # 给 save() 用
        self.vectorstore = FAISS.from_documents(children, self.embeddings)
        
        logger.info("FAISS index built with %d child documents", len(children))
    
    def save(self) -> None:
        """Persist the in-memory FAISS index to local index directory."""
        if self.vectorstore is None:
            raise ValueError("vectorstore is None, build or load before save")
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))
        logger.info("FAISS index saved to %s", self.index_dir)

        # 关键：同时保存 meta（用当前 children 数量）
        # 建议在 build() 里记录 self._last_children_count
        meta: dict[str, Any] = self._build_signature(self._last_children_count)
        self._save_meta(meta)
        logger.info("FAISS index saved with meta: %s", meta)

    def load(self, expected_children_count: int | None = None) -> bool:
        """Load FAISS index from local storage; return False if not found."""
        if not self.exists():
            logger.info("FAISS index not found at %s", self.index_dir)
            self.last_meta_match = None
            self.last_meta_reason = "index_files_missing"
            return False
        
        #expected signature
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
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded from %s", self.index_dir)
        return True

    def as_retriever(self, k: int) -> Any:
        """Expose the vector store as a similarity retriever with top-k setting."""
        if self.vectorstore is None:
            raise ValueError("vectorstore is None, build or load before save")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    @property
    def _meta_path(self) -> Path:
        return self.index_dir / "index.meta.json"
    
    def _build_signature(self, children_count: int) -> dict[str, Any]:
        """
        Build signature from current runtime config + data shape.
        """
        return {
            "source_dirs": sorted(self.cfg.source_dirs),
            "embedding_model": self.cfg.embedding_model,
            "embedding_dimensions": int(self.cfg.embedding_dimensions),
            "chunk_size": int(self.cfg.chunk_size), 
            "chunk_overlap": int(self.cfg.chunk_overlap),
            "children_count": int(children_count),
        }
    
    def _save_meta(self, meta: dict[str, Any]) -> None:
        """
        Persist meta signature alongside FAISS files.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def _load_meta(self) -> dict | None:
        """
        Load stored meta signature. Return None if missing/corrupted.
        """
        if not self._meta_path.exists():
            return None
        try:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    
    def _is_meta_match(self, expected_meta: dict) -> bool:
        """
        Compare current meta signature with expected signature.
        """
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