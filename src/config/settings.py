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
    text_model_provider: str = "glm_sync"
    glm_api_key: str | None = None
    glm_model: str = "glm-4-flashx-250414"
    glm_router_model: str | None = None
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_async_submit_path: str = "/async/chat/completions"
    glm_async_result_path: str = "/async-result/{id}"
    glm_poll_interval_s: float = 1.0
    glm_max_poll_s: float = 120.0
    glm_temperature: float = 1.0
    verbose: bool = False
    agent_runtime: str = "langgraph"
    agent_streaming: bool = True
    agent_use_langgraph_memory: bool = True
    agent_timeout_s: float = 90.0
    agent_request_timeout_s: float = 60.0
    agent_max_iterations: int = 3
    router_model: str = "qwen-max"
    hitl_action: str = "interrupt"  # "interrupt" | "warn" | "off"
    hitl_resume_default: str = "approve"
    elevenlabs_enabled: bool = False
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "Rachel"
    elevenlabs_tts_model: str = "eleven_multilingual_v2"
    elevenlabs_stt_model: str = "scribe_v1"
    elevenlabs_timeout_ms: int = 30000

    # 多模态
    multimodal_enabled: bool = False
    multimodal_provider: str = "dashscope"
    multimodal_model: str = "qwen3.6-plus"
    multimodal_api_key: str | None = None
    multimodal_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    multimodal_timeout_ms: int = 10000
    multimodal_max_image_bytes: int = 8 * 1024 * 1024
    multimodal_max_images_per_request: int = 3
    multimodal_allowed_mime: list[str] = field(default_factory=lambda: [
        "image/jpeg", "image/png", "image/webp",
    ])
    multimodal_assets_dir: str = "data/rag_assets/images"
    multimodal_uploads_dir: str = "data/uploads"
    multimodal_assets_url_prefix: str = "/static/rag_assets/images"
    multimodal_uploads_url_prefix: str = "/static/uploads"
    multimodal_min_image_bytes: int = 4096  # 跳过过小的图标

    # RAG
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
    vector_db_provider: str = "faiss"  # "faiss" | "qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_documents"
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
    rag_rerank_backend: str = "qwen"  # "qwen" | "crossencoder" | "none"
    rag_rerank_local_model: str = "BAAI/bge-reranker-v2-m3"
    rag_rerank_device: str = "cpu"    # "cpu" | "cuda" | "mps"
    postgres_uri: str | None = None
    enabled_skills: list[str] | None = None    # 新增
    guardrails_enabled: bool = True
    harness_enabled: bool = False
    harness_max_repair: int = 1
    supervisor_confidence_threshold: float = 0.6
    triage_confidence_threshold: float = 0.75
    image_analysis_enabled: bool = False
    hitl_enabled: bool = False
    mem0_enabled: bool = False
    mem0_api_key: str | None = None
    mem0_user_scope: str = "session"  # "session" | "global"

    agent_mode: str = "single"  # "single" | "multi" | "swarm"
    swarm_max_workers: int = 2
    swarm_timeout_s: float = 90.0
    complexity_threshold: float = 0.85
    
    # 网络搜索
    tavily_api_key: str | None = None
    tavily_enabled: bool = False


def load_settings() -> Settings:
    load_dotenv()

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    text_model_provider = os.getenv("TEXT_MODEL_PROVIDER", "glm_sync").strip().lower() or "glm_sync"
    if text_model_provider not in {"openai_compatible", "glm_async", "glm_sync"}:
        logger.warning("Invalid TEXT_MODEL_PROVIDER=%r, fallback to glm_sync", text_model_provider)
        text_model_provider = "glm_sync"
    glm_api_key = os.getenv("GLM_API_KEY", "").strip() or None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if text_model_provider in {"glm_async", "glm_sync"}:
        if not glm_api_key:
            raise ValueError(
                "Missing GLM_API_KEY. Please create a .env file based on .env.example."
            )
        api_key = api_key or glm_api_key
    elif not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Please create a .env file based on .env.example."
        )
    glm_model = os.getenv("GLM_MODEL", "glm-4-flashx-250414").strip() or "glm-4-flashx-250414"
    glm_router_model = os.getenv("GLM_ROUTER_MODEL", "").strip() or None
    glm_base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip() or "https://open.bigmodel.cn/api/paas/v4"
    glm_async_submit_path = os.getenv("GLM_ASYNC_SUBMIT_PATH", "/async/chat/completions").strip() or "/async/chat/completions"
    glm_async_result_path = os.getenv("GLM_ASYNC_RESULT_PATH", "/async-result/{id}").strip() or "/async-result/{id}"
    glm_poll_interval_s = _parse_float("GLM_POLL_INTERVAL_S", 1.0, 0.1, 10.0)
    glm_max_poll_s = _parse_float("GLM_MAX_POLL_S", 120.0, 1.0, 600.0)
    glm_temperature = _parse_float("GLM_TEMPERATURE", 1.0, 0.0, 2.0)
    verbose = os.getenv("AGENT_VERBOSE", "false").lower() in {"1", "true", "yes"}
    agent_use_langgraph_memory = _parse_bool("AGENT_USE_LANGGRAPH_MEMORY", True)
    agent_timeout_s = _parse_float("AGENT_TIMEOUT_S", 90.0, 5.0, 600.0)
    agent_request_timeout_s = _parse_float("AGENT_REQUEST_TIMEOUT_S", 60.0, 5.0, 300.0)
    agent_max_iterations = _parse_int("AGENT_MAX_ITERATIONS", 3, 1)
    router_model = os.getenv("ROUTER_MODEL", "qwen-max").strip() or "qwen-max"
    elevenlabs_enabled = _parse_bool("ELEVENLABS_ENABLED", False)
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "").strip() or None
    elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "Rachel").strip() or "Rachel"
    elevenlabs_tts_model = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2").strip() or "eleven_multilingual_v2"
    elevenlabs_stt_model = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1").strip() or "scribe_v1"
    elevenlabs_timeout_ms = _parse_int("ELEVENLABS_TIMEOUT_MS", 30000, 1000)

    multimodal_enabled = _parse_bool("MULTIMODAL_ENABLED", False)
    multimodal_provider = os.getenv("MULTIMODAL_PROVIDER", "dashscope").strip().lower() or "dashscope"
    if multimodal_provider not in {"dashscope"}:
        logger.warning("Invalid MULTIMODAL_PROVIDER=%r, fallback to dashscope", multimodal_provider)
        multimodal_provider = "dashscope"
    multimodal_model = os.getenv("MULTIMODAL_MODEL", "qwen3.6-plus").strip() or "qwen3.6-plus"
    multimodal_api_key = (
        os.getenv("MULTIMODAL_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or os.getenv("RAG_EMBEDDING_API_KEY", "").strip()
        or None
    )
    multimodal_base_url = os.getenv("MULTIMODAL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    multimodal_timeout_ms = _parse_int("MULTIMODAL_TIMEOUT_MS", 10000, 1000)
    multimodal_max_image_bytes = _parse_int("MULTIMODAL_MAX_IMAGE_BYTES", 8 * 1024 * 1024, 1024)
    multimodal_max_images_per_request = _parse_int("MULTIMODAL_MAX_IMAGES_PER_REQUEST", 3, 1)
    multimodal_allowed_mime = _parse_csv("MULTIMODAL_ALLOWED_MIME", "image/jpeg,image/png,image/webp")
    multimodal_assets_dir = os.getenv("MULTIMODAL_ASSETS_DIR", "data/rag_assets/images").strip() or "data/rag_assets/images"
    multimodal_uploads_dir = os.getenv("MULTIMODAL_UPLOADS_DIR", "data/uploads").strip() or "data/uploads"
    multimodal_assets_url_prefix = (
        os.getenv("MULTIMODAL_ASSETS_URL_PREFIX", "/static/rag_assets/images").strip()
        or "/static/rag_assets/images"
    )
    multimodal_uploads_url_prefix = (
        os.getenv("MULTIMODAL_UPLOADS_URL_PREFIX", "/static/uploads").strip()
        or "/static/uploads"
    )
    multimodal_min_image_bytes = _parse_int("MULTIMODAL_MIN_IMAGE_BYTES", 4096, 0)

    rag_source_dirs = _normalize_paths(_parse_csv("RAG_SOURCE_DIRS", ""))
    rag_embedding_api_key = _resolve_rag_embedding_api_key()

    rag_top_k = _parse_int("RAG_TOP_K", 4, 1)
    rag_retrieval_k = _parse_int("RAG_RETRIEVAL_K", 12, 1)
    vector_db_provider = os.getenv("VECTOR_DB_PROVIDER", "faiss").strip().lower() or "faiss"
    if vector_db_provider not in {"faiss", "qdrant"}:
        logger.warning("Invalid VECTOR_DB_PROVIDER=%r, fallback to faiss", vector_db_provider)
        vector_db_provider = "faiss"
    qdrant_url = os.getenv("QDRANT_URL", "").strip() or None
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "rag_documents").strip() or "rag_documents"
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

    postgres_uri = os.getenv("POSTGRES_URI", "").strip() or None

    enabled_skills_raw = _parse_csv("ENABLED_SKILLS", "")
    enabled_skills = enabled_skills_raw if enabled_skills_raw else None

    guardrails_enabled = _parse_bool("GUARDRAILS_ENABLED", True)
    harness_enabled = _parse_bool("HARNESS_ENABLED", False)
    harness_max_repair = _parse_int("HARNESS_MAX_REPAIR", 1, 0)
    supervisor_confidence_threshold = _parse_float("SUPERVISOR_CONFIDENCE_THRESHOLD", 0.6, 0.0, 1.0)
    triage_confidence_threshold = _parse_float("TRIAGE_CONFIDENCE_THRESHOLD", 0.75, 0.0, 1.0)
    image_analysis_enabled = _parse_bool("IMAGE_ANALYSIS_ENABLED", multimodal_enabled)
    hitl_enabled = _parse_bool("HITL_ENABLED", False)
    mem0_enabled = _parse_bool("MEM0_ENABLED", False)
    mem0_api_key = os.getenv("MEM0_API_KEY", "").strip() or None
    mem0_user_scope = os.getenv("MEM0_USER_SCOPE", "session").strip().lower() or "session"
    if mem0_user_scope not in {"session", "global"}:
        logger.warning("Invalid MEM0_USER_SCOPE=%r, fallback to session", mem0_user_scope)
        mem0_user_scope = "session"
    hitl_action = os.getenv("HITL_ACTION", "interrupt").strip().lower() or "interrupt"
    if hitl_action not in {"interrupt", "warn", "off"}:
        logger.warning("Invalid HITL_ACTION=%r, fallback to interrupt", hitl_action)
        hitl_action = "interrupt"
    hitl_resume_default = os.getenv("HITL_RESUME_DEFAULT", "approve").strip().lower() or "approve"

    agent_mode = os.getenv("AGENT_MODE", "single").strip().lower()
    if agent_mode not in {"single", "multi", "swarm"}:
        logger.warning("Invalid AGENT_MODE=%r, fallback to single", agent_mode)
        agent_mode = "single"
    rag_rerank_backend = os.getenv("RAG_RERANK_BACKEND", "qwen").strip().lower() or "qwen"
    if rag_rerank_backend not in {"qwen", "crossencoder", "none"}:
        logger.warning("Invalid RAG_RERANK_BACKEND=%r, fallback to qwen", rag_rerank_backend)
        rag_rerank_backend = "qwen"
    rag_rerank_local_model = os.getenv("RAG_RERANK_LOCAL_MODEL", "BAAI/bge-reranker-v2-m3").strip() or "BAAI/bge-reranker-v2-m3"
    rag_rerank_device = os.getenv("RAG_RERANK_DEVICE", "cpu").strip().lower() or "cpu"

    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip() or None
    tavily_enabled = _parse_bool("TAVILY_ENABLED", False)

    swarm_max_workers = _parse_int("SWARM_MAX_WORKERS", 2, 1)
    swarm_timeout_s = _parse_float("SWARM_TIMEOUT_S", 90.0, 1.0, 600.0)
    complexity_threshold = _parse_float("COMPLEXITY_THRESHOLD", 0.85, 0.0, 1.0)

    return Settings(
        openai_api_key=api_key,
        openai_model=model,
        openai_base_url=base_url,
        text_model_provider=text_model_provider,
        glm_api_key=glm_api_key,
        glm_model=glm_model,
        glm_router_model=glm_router_model,
        glm_base_url=glm_base_url,
        glm_async_submit_path=glm_async_submit_path,
        glm_async_result_path=glm_async_result_path,
        glm_poll_interval_s=glm_poll_interval_s,
        glm_max_poll_s=glm_max_poll_s,
        glm_temperature=glm_temperature,
        verbose=verbose,
        agent_runtime=_parse_runtime("AGENT_RUNTIME", "langgraph"),
        agent_streaming=_parse_bool("AGENT_STREAMING", True),
        agent_use_langgraph_memory=agent_use_langgraph_memory,
        agent_timeout_s=agent_timeout_s,
        agent_request_timeout_s=agent_request_timeout_s,
        agent_max_iterations=agent_max_iterations,
        router_model=router_model,
        elevenlabs_enabled=elevenlabs_enabled,
        elevenlabs_api_key=elevenlabs_api_key,
        elevenlabs_voice_id=elevenlabs_voice_id,
        elevenlabs_tts_model=elevenlabs_tts_model,
        elevenlabs_stt_model=elevenlabs_stt_model,
        elevenlabs_timeout_ms=elevenlabs_timeout_ms,
        multimodal_enabled=multimodal_enabled,
        multimodal_provider=multimodal_provider,
        multimodal_model=multimodal_model,
        multimodal_api_key=multimodal_api_key,
        multimodal_base_url=multimodal_base_url,
        multimodal_timeout_ms=multimodal_timeout_ms,
        multimodal_max_image_bytes=multimodal_max_image_bytes,
        multimodal_max_images_per_request=multimodal_max_images_per_request,
        multimodal_allowed_mime=multimodal_allowed_mime,
        multimodal_assets_dir=multimodal_assets_dir,
        multimodal_uploads_dir=multimodal_uploads_dir,
        multimodal_assets_url_prefix=multimodal_assets_url_prefix,
        multimodal_uploads_url_prefix=multimodal_uploads_url_prefix,
        multimodal_min_image_bytes=multimodal_min_image_bytes,
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
        vector_db_provider=vector_db_provider,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=qdrant_collection,
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
        postgres_uri=postgres_uri,
        enabled_skills=enabled_skills,
        agent_mode=agent_mode,
        guardrails_enabled=guardrails_enabled,
        harness_enabled=harness_enabled,
        harness_max_repair=harness_max_repair,
        supervisor_confidence_threshold=supervisor_confidence_threshold,
        triage_confidence_threshold=triage_confidence_threshold,
        image_analysis_enabled=image_analysis_enabled,
        hitl_enabled=hitl_enabled,
        mem0_enabled=mem0_enabled,
        mem0_api_key=mem0_api_key,
        mem0_user_scope=mem0_user_scope,
        hitl_action=hitl_action,
        hitl_resume_default=hitl_resume_default,
        
        rag_rerank_backend=rag_rerank_backend,
        rag_rerank_local_model=rag_rerank_local_model,
        rag_rerank_device=rag_rerank_device,

        tavily_api_key=tavily_api_key,
        tavily_enabled=tavily_enabled,

        swarm_max_workers=swarm_max_workers,
        swarm_timeout_s=swarm_timeout_s,
        complexity_threshold=complexity_threshold,

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
    if raw == "langgraph":
        return raw
    logger.warning("Invalid runtime env %s=%r, fallback to default=%s", name, raw, default)
    return default

def _parse_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%r, fallback to default=%s", name, raw, default)
        return default
    if value < minimum:
        logger.warning("Env %s=%s < minimum=%s, clamp to minimum", name, value, minimum)
        return minimum
    if value > maximum:
        logger.warning("Env %s=%s > maximum=%s, clamp to maximum", name, value, maximum)
        return maximum
    return value

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

