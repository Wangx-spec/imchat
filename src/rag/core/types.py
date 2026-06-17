from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGConfig:
    enabled: bool = False
    source_dirs: list[str] = field(default_factory=list)
    index_dir: str = "data/rag_index"
    chunk_overlap: int = 120
    top_k: int = 4
    retrieval_k: int = 12
    embedding_provider: str = "dashscope"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    vector_db_provider: str = "faiss"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_documents"
    chunk_size: int = 800
    rrf_k: int = 60
    rebuild: bool = False
    rerank_enabled: bool = False
    rerank_model: str = "qwen3-rerank"
    rerank_api_key: str | None = None
    rerank_endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    rerank_top_n: int = 8
    rerank_timeout_ms: int = 3000
    rerank_candidate_k: int = 40
    rag_query_plan_model: str = "qwen2.5-coder-7b-instruct"
    rag_query_plan_api_key: str | None = None
    rag_query_plan_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rag_query_plan_timeout_ms: int = 2000
    rag_query_plan_max_variants: int = 5
    rerank_backend: str = "qwen"
    rerank_local_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cpu"


@dataclass
class RetrievalResult:
    query: str
    parents: list[Any] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    query: str
    route: str
    answer: str
    sources: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    insufficient_info: bool = False
