from __future__ import annotations
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import Any
import logging
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

    def exists(self) -> bool:
        """Check whether a previously saved FAISS index exists on disk."""
        # FAISS.save_local 会生成 index.faiss 和 index.pkl
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def build(self, children: list[Document]) -> None:
        """Build a FAISS vector index from child documents in memory."""
        if not children:
            raise ValueError("children is empty, cannot build vector index")
        
        self.vectorstore = FAISS.from_documents(children, self.embeddings)
        logger.info("FAISS index built with %d child documents", len(children))
    
    def save(self) -> None:
        """Persist the in-memory FAISS index to local index directory."""
        if self.vectorstore is None:
            raise ValueError("vectorstore is None, build or load before save")
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))
        logger.info("FAISS index saved to %s", self.index_dir)

    def load(self) -> bool:
        """Load FAISS index from local storage; return False if not found."""
        if not self.exists():
            logger.info("FAISS index not found at %s", self.index_dir)
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