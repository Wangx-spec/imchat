import os
import logging
from pathlib import Path

from dataclasses import dataclass, field


from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class Settings:
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    verbose: bool = False
    agent_runtime: str = "langgraph"
    agent_streaming: bool = True
    agent_use_langgraph_memory: bool = True
    rag_enabled: bool = False
    rag_source_dirs: list[str] = field(default_factory=list)
    rag_index_dir: str = "data/rag_index"
    rag_top_k: int = 4
    rag_retrieval_k: int = 12
    rag_embedding_provider: str = "dashscope"
    rag_embedding_api_key: str | None = None
    rag_embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rag_embedding_model: str = "text-embedding-v4"
    rag_embedding_dimensions: int = 1024
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120
    rag_rrf_k: int = 60
    rag_rebuild: bool = False
    rag_rerank_enabled: bool = False
    rag_rerank_model: str = "qwen3-rerank"
    rag_rerank_api_key: str | None = None
    rag_rerank_endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    rag_rerank_top_n: int = 8
    rag_rerank_timeout_ms: int = 3000
    rag_rerank_candidate_k: int = 40
    rag_query_plan_model: str = "qwen2.5-coder-7b-instruct"
    rag_query_plan_api_key: str | None = None
    rag_query_plan_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rag_query_plan_timeout_ms: int = 2000
    rag_query_plan_max_variants: int = 5


def load_settings() -> Settings:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Please create a .env file based on .env.example."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    verbose = os.getenv("AGENT_VERBOSE", "false").lower() in {"1", "true", "yes"}
    agent_use_langgraph_memory = _parse_bool("AGENT_USE_LANGGRAPH_MEMORY", True)

    rag_source_dirs = _normalize_paths(_parse_csv("RAG_SOURCE_DIRS", ""))
    rag_embedding_api_key = _resolve_rag_embedding_api_key()

    rag_top_k = _parse_int("RAG_TOP_K", 4, 1)
    rag_retrieval_k = _parse_int("RAG_RETRIEVAL_K", 12, 1)
    rag_chunk_size = _parse_int("RAG_CHUNK_SIZE", 800, 100)
    rag_chunk_overlap = _parse_int("RAG_CHUNK_OVERLAP", 120, 0)
    rag_rrf_k = _parse_int("RAG_RRF_K", 60, 1)
    rag_embedding_dimensions = _parse_int("RAG_EMBEDDING_DIMENSIONS", 1024, 128)

    rag_rerank_enabled = _parse_bool("RAG_RERANK_ENABLED", False)
    rag_rerank_model = os.getenv("RAG_RERANK_MODEL", "qwen3-rerank").strip() or "qwen3-rerank"
    rag_rerank_api_key = (
        os.getenv("RAG_RERANK_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or None
    )
    rag_rerank_endpoint = os.getenv(
        "RAG_RERANK_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    ).strip()
    rag_rerank_top_n = _parse_int("RAG_RERANK_TOP_N", 8, 1)
    rag_rerank_timeout_ms = _parse_int("RAG_RERANK_TIMEOUT_MS", 3000, 500)
    rag_rerank_candidate_k = _parse_int("RAG_RERANK_CANDIDATE_K", 40, 5)
    rag_query_plan_model = os.getenv("RAG_QUERY_PLAN_MODEL", "qwen2.5-coder-7b-instruct").strip() or "qwen2.5-coder-7b-instruct"
    rag_query_plan_api_key = (
        os.getenv("RAG_QUERY_PLAN_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or None
    )
    rag_query_plan_base_url = os.getenv(
        "RAG_QUERY_PLAN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rag_query_plan_timeout_ms = _parse_int("RAG_QUERY_PLAN_TIMEOUT_MS", 2000, 500)
    rag_query_plan_max_variants = _parse_int("RAG_QUERY_PLAN_MAX_VARIANTS", 5, 1)

    return Settings(
        openai_api_key=api_key,
        openai_model=model,
        openai_base_url=base_url,
        verbose=verbose,
        agent_runtime=_parse_runtime("AGENT_RUNTIME", "langgraph"),
        agent_streaming=_parse_bool("AGENT_STREAMING", True),
        agent_use_langgraph_memory=agent_use_langgraph_memory,
        rag_enabled=_parse_bool("RAG_ENABLED", False),
        rag_source_dirs=rag_source_dirs,
        rag_index_dir=os.getenv("RAG_INDEX_DIR", "data/rag_index").strip() or "data/rag_index",
        rag_top_k=rag_top_k,
        rag_retrieval_k=rag_retrieval_k,
        rag_embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "dashscope").strip() or "dashscope",
        rag_embedding_api_key=rag_embedding_api_key,
        rag_embedding_base_url=os.getenv(
            "RAG_EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v4").strip() or "text-embedding-v4",
        rag_embedding_dimensions=rag_embedding_dimensions,
        rag_chunk_size=rag_chunk_size,
        rag_chunk_overlap=rag_chunk_overlap,
        rag_rrf_k=rag_rrf_k,
        rag_rebuild=_parse_bool("RAG_REBUILD", False),
        rag_rerank_enabled=rag_rerank_enabled,
        rag_rerank_model=rag_rerank_model,
        rag_rerank_api_key=rag_rerank_api_key,
        rag_rerank_endpoint=rag_rerank_endpoint,
        rag_rerank_top_n=rag_rerank_top_n,
        rag_rerank_timeout_ms=rag_rerank_timeout_ms,
        rag_rerank_candidate_k=rag_rerank_candidate_k,
        rag_query_plan_model=rag_query_plan_model,
        rag_query_plan_api_key=rag_query_plan_api_key,
        rag_query_plan_base_url=rag_query_plan_base_url,
        rag_query_plan_timeout_ms=rag_query_plan_timeout_ms,
        rag_query_plan_max_variants=rag_query_plan_max_variants,
    )

def _parse_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid bool env %s=%r, fallback to default=%s", name, raw, default)
    return default


def _parse_runtime(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"langchain", "langgraph"}:
        return raw
    logger.warning("Invalid runtime env %s=%r, fallback to default=%s", name, raw, default)
    return default

def _parse_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid int env %s=%r, fallback to default=%s", name, raw, default)
        return default
    if value < minimum:
        logger.warning("Env %s=%s < minimum=%s, clamp to minimum", name, value, minimum)
        return minimum
    return value

def _parse_csv(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default).strip()
    if not raw:
        return []
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]

def _normalize_paths(paths: list[str]) -> list[str]:
    normalized: list[str] = []
    for p in paths:
        p = (p or "").strip()
        if not p:
            continue
        try:
            normalized.append(str(Path(p).expanduser().resolve()))
        except Exception:
            normalized.append(p)
    return normalized

def _resolve_rag_embedding_api_key() -> str | None:
    # 优先专用 key
    key = os.getenv("RAG_EMBEDDING_API_KEY", "").strip()
    if key:
        return key
    # 回退 DashScope 通用 key
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    return None

