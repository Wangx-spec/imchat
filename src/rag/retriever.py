from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import json
import urllib.request

from rag.types import RetrievalResult


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
        self.vectorstore = vectorstore
        self.children = children
        self.parent_map = parent_map
        self.child_parent = child_parent
        self.rrf_k = rrf_k

        self.bm25 = BM25Retriever.from_documents(children)
        self.bm25.k = 5
        self.rerank_enabled = rerank_enabled
        self.rerank_model = rerank_model
        self.rerank_api_key = rerank_api_key
        self.rerank_endpoint = rerank_endpoint
        self.rerank_top_n = rerank_top_n
        self.rerank_timeout_ms = rerank_timeout_ms
        self.rerank_candidate_k = rerank_candidate_k

    def vector_search(self, query: str, k: int) -> list[Document]:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def bm25_search(self, query: str, k: int) -> list[Document]:
        self.bm25.k = k
        return self.bm25.invoke(query)

    def _doc_key(self, doc: Document) -> str:
        md = doc.metadata or {}
        cid = md.get("child_id")
        if cid:
            return str(cid)

        src = str(md.get("source", ""))
        idx = str(md.get("chunk_index", ""))
        if src or idx:
            return f"{src}::{idx}"

        return doc.page_content[:100]

    def _build_rerank_text(self, p: Document) -> str:
        md = p.metadata or {}
        title = str(md.get("title", "")).strip()
        source = str(md.get("source", "")).strip()
        snippet = p.page_content.strip().replace("\n", " ")[:600]
        return f"标题: {title}\n来源: {source}\n内容摘要: {snippet}"

    def _call_qwen_rerank(self, query: str, docs: list[str], top_n: int) -> tuple[list[int], list[float]]:
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
            d.metadata["rrf_score"] = scores[k]

        return fused

    def child_to_parent(self, fused_children: list[Document], top_k: int) -> list[Document]:
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
        out: list[str] = []
        seen: set[str] = set()
        for p in parents:
            s = str(p.metadata.get("source", ""))
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _extract_query_tokens(self, query: str) -> list[str]:
        q = (query or "").strip()
        if not q:
            return []

        # 去掉常见问句后缀，提取菜名核心词
        q_norm = q
        for pat in ["怎么做", "做法", "教程", "需要什么", "如何做", "怎么炒", "怎么煮", "怎么炖", "怎么烤"]:
            q_norm = q_norm.replace(pat, "")
        q_norm = re.sub(r"[？?！!。,.，；;：:\s]+", " ", q_norm).strip()

        tokens: list[str] = []
        if q_norm:
            tokens.append(q_norm)

        # 兼容“姜葱捞鸡 的做法”这类空格分段
        for part in q_norm.split(" "):
            p = part.strip()
            if len(p) >= 2:
                tokens.append(p)

        # 去重，长词优先
        uniq = []
        seen = set()
        for t in sorted(tokens, key=len, reverse=True):
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq[:6]

    def _score_parent_match(self, parent: Document, tokens: list[str], rank_idx: int) -> float:
        md = parent.metadata or {}
        title = str(md.get("title", "")).strip().lower()
        source = str(md.get("source", "")).strip().lower()
        source_name = source.replace("\\", "/").split("/")[-1].replace(".md", "")

        score = 0.0

        # 保留少量原始排序信息，避免完全打乱
        score += max(0.0, 0.05 - rank_idx * 0.005)

        for t in tokens:
            t_l = t.lower()
            if not t_l:
                continue

            if t_l == source_name:
                score += 2.5
            if t_l in title:
                score += 1.8
            if t_l in source:
                score += 1.2
            if title.startswith(t_l):
                score += 0.5
            if f"{t_l}的做法" == title:
                score += 1.0

        return score

    def _rerank_parents_by_query(self, parents: list[Document], query: str) -> tuple[list[Document], list[str], bool]:
        tokens = self._extract_query_tokens(query)
        if not parents or not tokens:
            return parents, tokens, False

        scored = []
        for i, p in enumerate(parents):
            s = self._score_parent_match(p, tokens, i)
            scored.append((s, i, p))

        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        reranked = [x[2] for x in scored]

        changed = any(a is not b for a, b in zip(parents, reranked))
        return reranked, tokens, changed

    def hybrid_search(self, query: str, retrieval_k: int, top_k: int) -> RetrievalResult:
        vector_docs = self.vector_search(query, retrieval_k)
        bm25_docs = self.bm25_search(query, retrieval_k)
        fused_children = self.rrf_fuse(vector_docs, bm25_docs)

        # 先放大 parent 候选池，再做 query-aware 重排，最后裁剪到 top_k
        candidate_parent_k = max(top_k * 8, 40)
        parents = self.child_to_parent(fused_children, top_k=candidate_parent_k)
        candidate_parent_count_before_trim = len(parents)
        top_titles_before_rerank = [str(p.metadata.get("title", "")) for p in parents[:5]]
        parents, tokens, rerank_applied = self._rerank_parents_by_query(parents, query)

        # qwen重排
        parents, rr_info = self._rerank_parents_by_qwen(query, parents)
        
        parents = parents[:top_k]

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
                "query_tokens": tokens,
                "rerank_applied": rerank_applied,
                "candidate_parent_k": candidate_parent_k,
                "candidate_parent_count_before_trim": candidate_parent_count_before_trim,
                "top_titles_before_rerank": top_titles_before_rerank,
                "top_titles": [str(p.metadata.get("title", "")) for p in parents[:3]],


                "qwen_rerank_enabled": self.rerank_enabled,
                "qwen_rerank_called": rr_info.get("rerank_called", False),
                "qwen_rerank_ok": rr_info.get("rerank_ok", False),
                "qwen_rerank_error": rr_info.get("rerank_error"),
                "qwen_rerank_scores": rr_info.get("rerank_scores", []),
                "qwen_rerank_model": self.rerank_model,
            },
        )