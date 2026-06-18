from __future__ import annotations

import logging
from typing import Any

from prompts.knowledge_base_prompt import build_blocked_answer
from rag.core.config import sanitize_rag_config, validate_rag_config
from rag.core.types import AnswerResult, RAGConfig, RetrievalResult
from rag.generation.generation_router import GenerationRouter
from rag.ingestion.chunking import ParentChildChunker
from rag.ingestion.data_loader import MarkdownDataLoader
from rag.retrieval.index_store import build_index_store
from rag.retrieval.query_planner import LLMQueryPlanner, QueryPlan
from rag.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RAGService:

    def __init__(self, cfg: RAGConfig, llm: Any | None = None, query_planner_llm: Any | None = None) -> None:
        # 配置先做一次sanitize，避免后续参数异常
        self.cfg = sanitize_rag_config(cfg)
        self.ready = False

        self.loader = MarkdownDataLoader()
        self.chunker = ParentChildChunker(self.cfg.chunk_size, self.cfg.chunk_overlap)
        self.index_store = build_index_store(self.cfg)
        self.router = GenerationRouter(llm=llm)
        self.query_planner_llm = query_planner_llm or llm
        self.query_planner: LLMQueryPlanner | None = None

        self.retriever: HybridRetriever | None = None

        # 运行期缓存，便于 debug/stats
        self.parents: list[Any] = []
        self.children: list[Any] = []
        self.parent_map: dict[str, Any] = {}
        self.child_parent: dict[str, str] = {}

        self._index_loaded = False
        self._index_rebuilt = False
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

        if self.query_planner_llm is not None:
            try:
                self.query_planner = LLMQueryPlanner(
                    llm=self.query_planner_llm,
                    max_variants=self.cfg.rag_query_plan_max_variants,
                )
            except Exception as exc:
                self.query_planner = None
                logger.warning("Query planner init failed, fallback to rule query expansion: %s", exc)
        elif self.cfg.rag_query_plan_api_key:
            try:
                self.query_planner = LLMQueryPlanner(
                    api_key=self.cfg.rag_query_plan_api_key,
                    base_url=self.cfg.rag_query_plan_base_url,
                    model=self.cfg.rag_query_plan_model,
                    timeout_ms=self.cfg.rag_query_plan_timeout_ms,
                    max_variants=self.cfg.rag_query_plan_max_variants,
                )
            except Exception as exc:
                self.query_planner = None
                logger.warning("Query planner init failed, fallback to rule query expansion: %s", exc)
        else:
            self.query_planner = None

        # Step3: index load/build
        self._index_loaded = False
        self._index_rebuilt = False  # 默认False
        self._index_meta_match = None
        self._index_meta_reason = None

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
            rerank_backend=self.cfg.rerank_backend,
            rerank_local_model=self.cfg.rerank_local_model,
            rerank_device=self.cfg.rerank_device,
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

        query_plan: QueryPlan | None = None
        if self.query_planner is not None:
            query_plan = self.query_planner.plan(query)
        
        ret = self.retriever.hybrid_search(
            query=query,
            retrieval_k=self.cfg.retrieval_k,
            top_k=self.cfg.top_k,
            query_plan=query_plan,
        )

        debug = dict(ret.debug or {})
        if query_plan is not None:
            debug["query_plan"] = {
                "used_llm": bool(query_plan.used_llm),
                "error": query_plan.error,
                "normalized_query": query_plan.normalized_query,
                "core_terms": query_plan.core_terms,
                "query_variants": query_plan.query_variants,
                "intent_hint": query_plan.intent_hint,
                "entities": query_plan.entities,
                "constraints": query_plan.constraints,
                "confidence": query_plan.confidence,
            }
        else:
            debug["query_plan"] = None

        return RetrievalResult(
            query=ret.query,
            parents=ret.parents,
            sources=ret.sources,
            debug=debug,
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
            debug = dict(ret.debug or {})
            confidence_score = float(debug.get("confidence_score", 0.0) or 0.0)
            is_confident = bool(debug.get("is_confident", False))
            # 兼容旧检索器：尚未输出新字段时回退到 exact_match_hit
            if "is_confident" not in debug:
                is_confident = bool(debug.get("exact_match_hit", False))

            direct_hit_count = int(debug.get("direct_hit_count", 0) or 0)
            top_titles = [str(x).lower() for x in (debug.get("top_titles") or [])[:3]]
            q_norm = str(debug.get("query_normalized", rewritten or query)).lower()
            semantic_overlap = any(t and (t in q_norm or q_norm in t) for t in top_titles)

            # 医学知识类问题下，detail/general/list 都应做低置信保护，
            # 避免旧路由把“有哪些表现/特征”误分到 list 后直接生成答非所问内容。
            confidence_threshold = 0.45
            has_strong_evidence = is_confident or direct_hit_count > 0 or semantic_overlap
            q_lower = (query or "").strip().lower()
            medical_query_hints = [
                "mri", "ct", "dwi", "adc", "影像", "征象", "表现", "特征",
                "病理", "诊断", "鉴别", "分级", "分型", "指南", "评估",
                "胶质", "肿瘤", "脑", "cns", "who", "rano",
            ]
            is_medical_knowledge_query = any(x in q_lower for x in medical_query_hints)
            should_block_low_confidence = (
                (route == "detail")
                or (route in {"general", "list"} and is_medical_knowledge_query)
            )

            if should_block_low_confidence and (confidence_score < confidence_threshold) and (not has_strong_evidence):
                top_sources = ret.sources[:3]
                source_lines = "\n".join([f"- {s}" for s in top_sources]) if top_sources else "- 无"
                blocked_answer = build_blocked_answer(source_lines)
                debug["low_confidence_blocked"] = True
                debug["insufficient_info"] = True
                debug["insufficient_reason"] = "low_confidence"
                logger.info(
                    "[ANSWER_GUARD] query=%r route=%s should_block=%s confidence_score=%.3f is_confident=%s direct_hit_count=%s semantic_overlap=%s low_confidence_blocked=%s variant_queries=%s direct_hit_titles=%s",
                    query,
                    route,
                    should_block_low_confidence,
                    confidence_score,
                    is_confident,
                    direct_hit_count,
                    semantic_overlap,
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
                    insufficient_info=True
                )

            logger.info(
                "[ANSWER_GUARD] query=%r route=%s should_block=%s confidence_score=%.3f is_confident=%s direct_hit_count=%s semantic_overlap=%s low_confidence_blocked=%s variant_queries=%s direct_hit_titles=%s",
                query,
                route,
                should_block_low_confidence,
                confidence_score,
                is_confident,
                direct_hit_count,
                semantic_overlap,
                False,
                debug.get("variant_queries", []),
                debug.get("direct_hit_titles", []),
            )
            answer_text = self.router.build_answer(query, route, ret.parents)
            insufficient_info = self.router.is_insufficient_answer(answer_text)
            debug["insufficient_info"] = insufficient_info

            return AnswerResult(
                query=query, 
                route=route, 
                answer=answer_text, 
                sources=ret.sources, 
                debug=debug, 
                insufficient_info=insufficient_info
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
