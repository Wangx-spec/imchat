from __future__ import annotations

from dataclasses import replace

from config.settings import Settings
from rag.types import RAGConfig


def build_rag_config(settings: Settings) -> RAGConfig:
    """
    1:1 映射 Settings -> RAGConfig（不做复杂逻辑）
    """
    cfg = RAGConfig(
        enabled=settings.rag_enabled,
        source_dirs=settings.rag_source_dirs,
        index_dir=settings.rag_index_dir,
        top_k=settings.rag_top_k,
        retrieval_k=settings.rag_retrieval_k,
        embedding_provider=settings.rag_embedding_provider,
        embedding_api_key=settings.rag_embedding_api_key,
        embedding_base_url=settings.rag_embedding_base_url,
        embedding_model=settings.rag_embedding_model,
        embedding_dimensions=settings.rag_embedding_dimensions,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        rrf_k=settings.rag_rrf_k,
        rebuild=settings.rag_rebuild,
    )
    return cfg


def sanitize_rag_config(cfg: RAGConfig) -> RAGConfig:
    """
    只修正“可自动修正”的配置，不抛异常。
    """
    # 用 replace 创建新对象，避免原地修改带来的副作用
    out = replace(cfg)

    if out.top_k <= 0:
        out.top_k = 1

    if out.retrieval_k < out.top_k:
        out.retrieval_k = out.top_k

    if out.chunk_overlap >= out.chunk_size:
        out.chunk_overlap = max(0, out.chunk_size // 5)

    return out


def validate_rag_config(cfg: RAGConfig) -> None:
    """
    对不可接受配置做硬校验，失败抛 ValueError。
    """
    if not cfg.index_dir or not cfg.index_dir.strip():
        raise ValueError("index_dir is required")

    if not cfg.embedding_provider or not cfg.embedding_provider.strip():
        raise ValueError("embedding_provider is required")

    if not cfg.embedding_model or not cfg.embedding_model.strip():
        raise ValueError("embedding_model is required")

    if not cfg.embedding_base_url or not cfg.embedding_base_url.strip():
        raise ValueError("embedding_base_url is required")

    if cfg.enabled and (not cfg.embedding_api_key or not cfg.embedding_api_key.strip()):
        raise ValueError("embedding_api_key is required when rag is enabled")

    if cfg.embedding_dimensions < 128:
        raise ValueError("embedding_dimensions must be >= 128")

    if cfg.top_k < 1:
        raise ValueError("top_k must be >= 1")

    if cfg.retrieval_k < cfg.top_k:
        raise ValueError("retrieval_k must be >= top_k")

    if cfg.chunk_size < 100:
        raise ValueError("chunk_size must be >= 100")

    if cfg.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")