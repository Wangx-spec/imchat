from __future__ import annotations

import logging
from typing import Any

from rag.types import RAGConfig, RetrievalResult, AnswerResult
from rag.config import sanitize_rag_config, validate_rag_config
from rag.data_loader import MarkdownDataLoader
from rag.chunking import ParentChildChunker
from rag.index_store import LocalFAISSIndexStore
from rag.retriever import HybridRetriever
from rag.generation_router import GenerationRouter

logger = logging.getLogger(__name__)

class RAGService:

    def __init__(self, cfg: RAGConfig) -> None:
        # 配置先做一次sanitize，避免后续参数异常
        self.cfg = sanitize_rag_config(cfg)
        self.ready = False

        self.loader = MarkdownDataLoader()
        self.chunker = ParentChildChunker(self.cfg.chunk_size, self.cfg.chunk_overlap)
        self.index_store = LocalFAISSIndexStore(self.cfg)
        self.router = GenerationRouter()

        self.retriever: HybridRetriever | None = None

        # 运行期缓存，便于 debug/stats
        self.parents: list[Any] = []
        self.children: list[Any] = []
        self.parent_map: dict[str, Any] = {}
        self.child_parent: dict[str, str] = {}
        
        self._index_loaded = False
        self._index_rebuilt = True # 默认False
        self._index_meta_match = None  # 可选：True/False/None
        self._index_meta_reason = None


    def initialize(self, force_rebuild: bool = False) -> None:
        if not self.cfg.enabled:
            self.ready = False
            logger.info("RAG disabled by config")
            return

        validate_rag_config(self.cfg)

        # Step2: load + chunk
        self.parents = self.loader.load_documents(self.cfg.source_dirs)
        if not self.parents:
            raise RuntimeError("No parent documents loaded from source_dirs")

        self.children, self.parent_map, self.child_parent = self.chunker.build_parent_child(self.parents)
        if not self.children:
            raise RuntimeError("No child chunks generated")

        # Step3: index load/build
        self._index_loaded = False
        self._index_rebuilt = False # 默认False         
        self._index_meta_match = None
        self._index_meta_reason = None

        # loaded = False
        if not (force_rebuild or self.cfg.rebuild):
            # 关键：传 expected_children_count 给 load 做签名校验
            self._index_loaded = self.index_store.load(expected_children_count=len(self.children))
            self._index_meta_match = self.index_store.last_meta_match
            self._index_meta_reason = self.index_store.last_meta_reason

        if not self._index_loaded:
            self.index_store.build(self.children)
            self.index_store.save()
            self._index_rebuilt = True
            self._index_meta_match = self.index_store.last_meta_match
            self._index_meta_reason = self.index_store.last_meta_reason

        # Step3: retriever init
        self.retriever = HybridRetriever(
            vectorstore=self.index_store.vectorstore,
            children=self.children,
            parent_map=self.parent_map,
            child_parent=self.child_parent,
            rrf_k=self.cfg.rrf_k,
            rerank_enabled=self.cfg.rerank_enabled,
            rerank_model=self.cfg.rerank_model,
            rerank_api_key=self.cfg.rerank_api_key,
            rerank_endpoint=self.cfg.rerank_endpoint,
            rerank_top_n=self.cfg.rerank_top_n,
            rerank_timeout_ms=self.cfg.rerank_timeout_ms,
            rerank_candidate_k=self.cfg.rerank_candidate_k,
        )

        self.ready = True
        logger.info(
            "RAG ready: parents=%d children=%d index_loaded=%s index_rebuilt=%s",
            len(self.parents),
            len(self.children),
            self._index_loaded,
            self._index_rebuilt,
        )
    
    def is_ready(self) -> bool:
        return self.ready

    def retrieve(self, query: str) -> RetrievalResult:
        if not self.cfg.enabled:
            return RetrievalResult(
                query=query,
                parents=[],
                sources=[],
                debug={"error": "disabled"},
            )
        if not self.ready or self.retriever is None:
            return RetrievalResult(
                query=query,
                parents=[],
                sources=[],
                debug={"error": "not_ready"},
            )
        return self.retriever.hybrid_search(
            query=query,
            retrieval_k=self.cfg.retrieval_k,
            top_k=self.cfg.top_k,
        )

    def answer(self, query: str) -> AnswerResult:
        if not self.cfg.enabled:
            return AnswerResult(query=query, route="disabled", answer="RAG 未启用。")
        if not self.ready:
            return AnswerResult(query=query, route="not_ready", answer="RAG 尚未初始化。")
        try:
            route = self.router.route_query(query)
            rewritten = self.router.rewrite_query(query, route)
            ret = self.retrieve(rewritten)
            exact_hit = bool((ret.debug or {}).get("exact_match_hit", False))

            # 低置信度门控：detail 场景没有明确命中时，避免输出“像正确答案”的幻觉内容
            if route == "detail" and not exact_hit:
                top_sources = ret.sources[:3]
                source_lines = "\n".join([f"- {s}" for s in top_sources]) if top_sources else "- 无"
                blocked_answer = (
                    "我没有在知识库中精确命中到该问题的目标条目，暂时不输出详细步骤，"
                    "以避免给出不可靠内容。你可以换一个更具体的问法（例如完整菜名/文档标题）。\n\n"
                    f"当前可参考来源：\n{source_lines}"
                )
                debug = dict(ret.debug or {})
                debug["low_confidence_blocked"] = True
                logger.info(
                    "[ANSWER_GUARD] query=%r route=%s exact_match_hit=%s low_confidence_blocked=%s variant_queries=%s direct_hit_titles=%s",
                    query,
                    route,
                    exact_hit,
                    True,
                    debug.get("variant_queries", []),
                    debug.get("direct_hit_titles", []),
                )
                return AnswerResult(
                    query=query,
                    route=route,
                    answer=blocked_answer,
                    sources=ret.sources,
                    debug=debug,
                )

            logger.info(
                "[ANSWER_GUARD] query=%r route=%s exact_match_hit=%s low_confidence_blocked=%s variant_queries=%s direct_hit_titles=%s",
                query,
                route,
                exact_hit,
                False,
                (ret.debug or {}).get("variant_queries", []),
                (ret.debug or {}).get("direct_hit_titles", []),
            )
            answer_text = self.router.build_answer(query, route, ret.parents)
            return AnswerResult(
                query=query,
                route=route,
                answer=answer_text,
                sources=ret.sources,
                debug=ret.debug,
            )
        except Exception as exc:
            logger.exception("RAG answer failed: %s", exc)
            return AnswerResult(
                query=query,
                route="error",
                answer=f"RAG 处理失败：{exc}",
                debug={"error": str(exc)},
            )

    def stats(self) -> dict:
        return {
            "enabled": self.cfg.enabled,
            "ready": self.ready,
            "source_dirs": self.cfg.source_dirs,
            "index_dir": self.cfg.index_dir,
            "parent_count": len(self.parents),
            "child_count": len(self.children),
            "retriever_ready": self.retriever is not None,
            "index_loaded": self._index_loaded,
            "index_rebuilt": self._index_rebuilt,
            "index_meta_match": self._index_meta_match,
            "index_meta_reason": self._index_meta_reason,
        }
    
    def route_query(self, query: str) -> str:
        q = query.strip().lower()
        if any(x in q for x in ["推荐", "有哪些", "来几道", "列出"]):
            return "list"
        if any(x in q for x in ["怎么做", "做法", "步骤", "需要什么"]):
            return "detail"
        return "general"