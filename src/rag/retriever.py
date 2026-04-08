from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import json
import urllib.request
from typing import Any
from rag.query_planner import QueryPlan

from rag.types import RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vectorstore: Any,
        children: list[Document],
        parent_map: dict[str, Document],
        child_parent: dict[str, str],
        rrf_k: int = 60,
        rerank_enabled: bool = False,
        rerank_model: str = "qwen3-rerank",
        rerank_api_key: str | None = None,
        rerank_endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        rerank_top_n: int = 8,
        rerank_timeout_ms: int = 3000,
        rerank_candidate_k: int = 40,
    ) -> None:
        """初始化混合检索器及其可选的 Qwen 重排参数。"""
        self.vectorstore = vectorstore
        self.children = children
        self.parent_map = parent_map
        self.child_parent = child_parent
        self.rrf_k = rrf_k

        self.bm25 = BM25Retriever.from_documents(
            children,
            preprocess_func=self._tokenize_for_sparse,
        )
        self.bm25.k = 5
        self.rerank_enabled = rerank_enabled
        self.rerank_model = rerank_model
        self.rerank_api_key = rerank_api_key
        self.rerank_endpoint = rerank_endpoint
        self.rerank_top_n = rerank_top_n
        self.rerank_timeout_ms = rerank_timeout_ms
        self.rerank_candidate_k = rerank_candidate_k

    def vector_search(self, query: str, k: int) -> list[Document]:
        """执行向量检索并返回前 k 个子文档。"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def bm25_search(self, query: str, k: int) -> list[Document]:
        """执行 BM25 稀疏检索并返回前 k 个子文档。"""
        self.bm25.k = k
        return self.bm25.invoke(query)

    def _tokenize_for_sparse(self, text: str) -> list[str]:
        """为稀疏检索分词：英文词、中文单字和中文双字。"""
        t = (text or "").lower().strip()
        if not t:
            return []

        words = re.findall(r"[a-z0-9_]+", t)
        zh = re.findall(r"[\u4e00-\u9fff]", t)
        bigrams = ["".join(zh[i : i + 2]) for i in range(len(zh) - 1)]
        return words + zh + bigrams
    
    def _variants_from_plan(self, query: str, query_plan: QueryPlan | None) -> list[str]:
        if query_plan and query_plan.query_variants:
            out: list[str] = []
            seen: set[str] = set()
            for v in query_plan.query_variants:
                s = (v or "").strip()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            if out:
                return out
        q = (query or "").strip()
        return [q] if q else []
    
    def _fallback_parent_recall(
        self,
        terms: list[str],
        limit: int,
        min_overlap: int = 1,
    ) -> list[Document]:
        if not terms or limit <= 0:
            return []
        norm_terms = [self._normalize_text(t) for t in terms if self._normalize_text(t)]
        if not norm_terms:
            return []
        scored: list[tuple[float, Document]] = []
        for p in self.parent_map.values():
            md = p.metadata or {}
            hay = self._normalize_text(
                f"{md.get('title', '')} {md.get('source', '')} {p.page_content[:1200]}"
            )
            if not hay:
                continue
            hit = sum(1 for t in norm_terms if t and t in hay)
            if hit < min_overlap:
                continue
            # 简单通用分：覆盖率 + 命中数
            coverage = hit / max(1, len(norm_terms))
            score = coverage + hit * 0.05
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:limit]]

    def _score_evidence(
        self,
        query: str,
        query_plan: QueryPlan | None,
        parents: list[Document],
        rr_info: dict[str, Any],
    ) -> tuple[float, bool, list[str], dict[str, Any]]:
        reasons: list[str] = []
        details: dict[str, Any] = {}

        if not parents:
            return 0.0, False, ["no_parent_candidates"], {"term_coverage": 0.0}

        score = 0.0
        score += 0.20  # 有候选父文档基础分
        reasons.append("has_parent_candidates")

        top_docs = parents[:3]
        hay = " ".join(
            self._normalize_text(
                f"{(d.metadata or {}).get('title', '')} {(d.metadata or {}).get('source', '')} {d.page_content[:600]}"
            )
            for d in top_docs
        )

        terms: list[str] = []
        if query_plan:
            terms.extend(query_plan.core_terms or [])
            terms.extend(query_plan.entities or [])
        if not terms:
            terms = [query]

        norm_terms = [self._normalize_text(t) for t in terms if self._normalize_text(t)]
        hit_terms = [t for t in norm_terms if t in hay]
        coverage = len(hit_terms) / max(1, len(norm_terms))
        details["term_coverage"] = coverage
        details["term_hits"] = hit_terms[:8]

        # term 覆盖贡献
        score += min(0.45, coverage * 0.45)
        if coverage >= 0.5:
            reasons.append("term_coverage_good")
        elif coverage > 0:
            reasons.append("term_coverage_partial")
        else:
            reasons.append("term_coverage_none")

        # rerank 贡献（若可用）
        if rr_info.get("rerank_ok"):
            raw_scores = rr_info.get("rerank_scores") or []
            top_rr = float(raw_scores[0]) if raw_scores else 0.0
            top_rr = max(0.0, min(1.0, top_rr))
            score += top_rr * 0.30
            details["top_rerank_score"] = top_rr
            reasons.append("qwen_rerank_ok")
        else:
            reasons.append("qwen_rerank_unavailable")

        # planner 成功微加分
        if query_plan and query_plan.used_llm and not query_plan.error:
            score += 0.05
            reasons.append("query_plan_used_llm")

        score = max(0.0, min(1.0, score))
        is_confident = score >= 0.45
        return score, is_confident, reasons, details

    def _normalize_text(self, text: str) -> str:
        t = (text or "").lower()
        t = re.sub(r"[^\w\u4e00-\u9fff]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _tokenize_generic(self, text: str) -> list[str]:
        t = self._normalize_text(text)
        if not t:
            return []
        # 英文术语/编号 + 中文单字（可覆盖多领域）
        words = re.findall(r"[a-z0-9._-]+", t)
        zh = re.findall(r"[\u4e00-\u9fff]", t)
        bigrams = ["".join(zh[i:i+2]) for i in range(len(zh) - 1)]
        return words + zh + bigrams

    def _doc_key(self, doc: Document) -> str:
        """生成子文档去重键，优先使用 child_id。"""
        md = doc.metadata or {}
        cid = md.get("child_id")
        if cid:
            return str(cid)

        src = str(md.get("source", ""))
        idx = str(md.get("chunk_index", ""))
        if src or idx:
            return f"{src}::{idx}"

        return doc.page_content[:100]

    def _parent_key(self, doc: Document) -> str:
        """生成父文档去重键，优先使用 parent_id/source/title。"""
        md = doc.metadata or {}
        pid = str(md.get("parent_id", "")).strip()
        if pid:
            return pid
        source = str(md.get("source", "")).strip()
        if source:
            return source
        title = str(md.get("title", "")).strip()
        if title:
            return title
        return doc.page_content[:120]

    def _build_rerank_text(self, p: Document) -> str:
        """将父文档构造成重排模型可读的文本条目。"""
        md = p.metadata or {}
        title = str(md.get("title", "")).strip()
        source = str(md.get("source", "")).strip()
        snippet = p.page_content.strip().replace("\n", " ")[:600]
        return f"标题: {title}\n来源: {source}\n内容摘要: {snippet}"

    def _call_qwen_rerank(self, query: str, docs: list[str], top_n: int) -> tuple[list[int], list[float]]:
        """调用 Qwen 重排接口并返回候选索引与相关性分数。"""
        payload = {
            "model": self.rerank_model,
            "input": {"query": query, "documents": docs},
            "parameters": {"return_documents": True, "top_n": top_n},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.rerank_endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.rerank_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        timeout_sec = max(0.5, self.rerank_timeout_ms / 1000.0)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = data.get("output", {}).get("results") or data.get("results") or []
        idxs, scores = [], []
        for r in results:
            if "index" in r:
                idxs.append(int(r["index"]))
                scores.append(float(r.get("relevance_score", 0.0)))
        return idxs, scores
    
    def _rerank_parents_by_qwen(self, query: str, parents: list[Document]) -> tuple[list[Document], dict[str, Any]]:
        """对父文档列表执行 Qwen 重排，失败时回退原顺序并返回状态信息。"""
        info = {"rerank_called": False, "rerank_ok": False, "rerank_error": None, "rerank_scores": []}

        if not self.rerank_enabled:
            info["rerank_error"] = "disabled"
            return parents, info
        if not self.rerank_api_key:
            info["rerank_error"] = "missing_api_key"
            return parents, info
        if not parents:
            info["rerank_error"] = "empty_candidates"
            return parents, info

        candidates = parents[: max(self.rerank_candidate_k, self.rerank_top_n)]
        docs = [self._build_rerank_text(p) for p in candidates]
        top_n = min(len(candidates), max(1, self.rerank_top_n))

        info["rerank_called"] = True
        try:
            idxs, scores = self._call_qwen_rerank(query, docs, top_n)
            if not idxs:
                info["rerank_error"] = "empty_rerank_result"
                return parents, info

            picked = [candidates[i] for i in idxs if 0 <= i < len(candidates)]
            picked_ids = {id(p) for p in picked}
            tail = [p for p in parents if id(p) not in picked_ids]
            reranked = picked + tail

            info["rerank_ok"] = True
            info["rerank_scores"] = scores
            return reranked, info
        except Exception as exc:
            info["rerank_error"] = f"{type(exc).__name__}: {exc}"
            return parents, info

    def rrf_fuse(self, vector_docs: list[Document], bm25_docs: list[Document]) -> list[Document]:
        """使用 RRF 融合向量与 BM25 的子文档排序结果。"""
        scores: dict[str, float] = {}
        doc_by_key: dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            key = self._doc_key(doc)
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            key = self._doc_key(doc)
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        fused = [doc_by_key[k] for k in ranked_keys]

        for d in fused:
            k = self._doc_key(d)
            if d.metadata is None:
                d.metadata = {}
            d.metadata["rrf_score"] = scores[k]

        return fused

    def child_to_parent(self, fused_children: list[Document], top_k: int) -> list[Document]:
        """将子文档结果映射为父文档并按首次命中顺序截断到 top_k。"""
        parents: list[Document] = []
        seen_parent: set[str] = set()

        for child in fused_children:
            cid = str(child.metadata.get("child_id", ""))
            pid = child.metadata.get("parent_id") or self.child_parent.get(cid)
            if not pid or pid in seen_parent:
                continue

            parent_doc = self.parent_map.get(pid)
            if parent_doc is None:
                continue

            seen_parent.add(pid)
            parents.append(parent_doc)

            if len(parents) >= top_k:
                break

        return parents

    def _collect_sources(self, parents: list[Document]) -> list[str]:
        """提取父文档来源字段并保持去重后的顺序。"""
        out: list[str] = []
        seen: set[str] = set()
        for p in parents:
            s = str(p.metadata.get("source", ""))
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _direct_parent_lexical_recall(self, query: str, limit: int, min_score: float = 2.5) -> list[Document]:
        """基于标题/来源/内容的词法匹配直接召回父文档。"""
        q = (query or "").strip().lower()
        if not q or limit <= 0:
            return []

        q_tokens = set(self._tokenize_for_sparse(q))
        hits: list[tuple[float, Document]] = []

        for p in self.parent_map.values():
            md = p.metadata or {}
            title = str(md.get("title", "")).lower()
            source = str(md.get("source", "")).lower()
            snippet = p.page_content[:1200].lower()

            score = 0.0

            # 强匹配：核心短语
            if core and len(core) >= 3:
                if core in title:
                    score += 4.0
                if core in source:
                    score += 3.0
                if core in snippet:
                    score += 2.0

            # 弱匹配：完整 query
            if q in title:
                score += 1.5
            if q in source:
                score += 1.0
            if q in snippet:
                score += 0.8

            # token overlap
            d_tokens = set(self._tokenize_for_sparse(f"{title} {source} {snippet[:300]}"))
            overlap = len(q_tokens & d_tokens)
            score += min(overlap * 0.08, 1.2)

            # 关键：阈值过滤，避免“蚂蚁上树/蒜蓉虾”这类弱相关被拉进来
            if score >= min_score:
                hits.append((score, p))

        hits.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in hits[:limit]]


    def _merge_unique_parents(self, primary: list[Document], extra: list[Document]) -> list[Document]:
        """按顺序合并两组父文档并基于父文档键去重。"""
        out: list[Document] = []
        seen: set[str] = set()
        for p in primary + extra:
            k = self._parent_key(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        return out

    def _compute_exact_match_hit(
        self,
        original_query: str,
        variants: list[str],
        core_terms: list[str],
        parents: list[Document],
    ) -> bool:
        """判断候选父文档中是否命中核心实体词。"""
        if not parents:
            return False

        terms: list[str] = []
        for t in core_terms:
            t = (t or "").strip().lower()
            if len(t) >= 2:
                terms.append(t)

        if not terms:
            return False

        for p in parents:
            md = p.metadata or {}
            title = str(md.get("title", "")).lower()
            source = str(md.get("source", "")).lower()
            snippet = p.page_content[:800].lower()
            for t in terms:
                if t in title or t in source or t in snippet:
                    return True
        return False

    def hybrid_search(
        self,
        query: str,
        retrieval_k: int,
        top_k: int,
        query_plan: QueryPlan | None = None,
    ) -> RetrievalResult:
        normalized_query = (query_plan.normalized_query if query_plan else query) or query
        variants = self._variants_from_plan(normalized_query, query_plan)

        if not variants:
            variants = [normalized_query]

        k_per = max(8, retrieval_k // max(1, len(variants)))
        all_vector_docs: list[Document] = []
        all_bm25_docs: list[Document] = []

        for v in variants:
            all_vector_docs.extend(self.vector_search(v, k_per))
            all_bm25_docs.extend(self.bm25_search(v, k_per))

        vector_docs = all_vector_docs
        bm25_docs = all_bm25_docs
        fused_children = self.rrf_fuse(vector_docs, bm25_docs)

        candidate_parent_k = max(top_k * 8, self.rerank_candidate_k, 40)
        parents_from_children = self.child_to_parent(fused_children, top_k=candidate_parent_k)

        # fallback only: 使用 planner 的 terms/entities，不做领域规则
        fallback_terms: list[str] = []
        if query_plan:
            fallback_terms.extend(query_plan.core_terms or [])
            fallback_terms.extend(query_plan.entities or [])
        if not fallback_terms:
            fallback_terms = [normalized_query]

        direct_hits = self._fallback_parent_recall(
            terms=fallback_terms,
            limit=max(1, candidate_parent_k // 3),
            min_overlap=1,
        )

        parents = self._merge_unique_parents(parents_from_children, direct_hits)
        candidate_parent_count_before_trim = len(parents)
        top_titles_before_rerank = [str((p.metadata or {}).get("title", "")) for p in parents[:5]]

        # 去掉规则重排，仅保留 qwen rerank
        parents, rr_info = self._rerank_parents_by_qwen(normalized_query, parents)

        confidence_score, is_confident, confidence_reasons, confidence_details = self._score_evidence(
            query=query,
            query_plan=query_plan,
            parents=parents,
            rr_info=rr_info,
        )

        parents = parents[:top_k]

        logger.info(
            "[RETRIEVE_DEBUG] query=%r variants=%s confidence_score=%.3f is_confident=%s",
            query,
            variants,
            confidence_score,
            is_confident,
        )

        # 兼容旧 service：先保留 exact_match_hit（映射为 is_confident）
        return RetrievalResult(
            query=query,
            parents=parents,
            sources=self._collect_sources(parents),
            debug={
                "vector_hits": len(vector_docs),
                "bm25_hits": len(bm25_docs),
                "fused_hits": len(fused_children),
                "parent_hits": len(parents),
                "rrf_k": self.rrf_k,
                "variant_queries": variants,
                "query_normalized": normalized_query,
                "query_core_terms": (query_plan.core_terms if query_plan else []),
                "query_plan_used_llm": bool(query_plan.used_llm) if query_plan else False,
                "query_plan_error": query_plan.error if query_plan else None,
                "candidate_parent_k": candidate_parent_k,
                "candidate_parent_count_before_trim": candidate_parent_count_before_trim,
                "direct_hit_count": len(direct_hits),
                "direct_hit_titles": [str((p.metadata or {}).get("title", "")) for p in direct_hits[:5]],
                "top_titles_before_rerank": top_titles_before_rerank,
                "top_titles": [str((p.metadata or {}).get("title", "")) for p in parents[:3]],
                "qwen_rerank_enabled": self.rerank_enabled,
                "qwen_rerank_called": rr_info.get("rerank_called", False),
                "qwen_rerank_ok": rr_info.get("rerank_ok", False),
                "qwen_rerank_error": rr_info.get("rerank_error"),
                "qwen_rerank_scores": rr_info.get("rerank_scores", []),
                "qwen_rerank_model": self.rerank_model,
                # 新证据判定
                "confidence_score": confidence_score,
                "is_confident": is_confident,
                "confidence_reasons": confidence_reasons,
                "confidence_details": confidence_details,
                # 兼容字段（过渡）
                "exact_match_hit": is_confident,
            },
        )