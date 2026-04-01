from __future__ import annotations
from typing import Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from rag.types import RetrievalResult


class HybridRetriever:

    def __init__(
        self,
        vectorstore: Any,
        children: list[Document],
        parent_map: dict[str, Document],
        child_parent: dict[str, str],
        rrf_k: int = 60
    ) -> None:
        self.vectorstore = vectorstore
        self.children = children
        self.parent_map = parent_map
        self.child_parent = child_parent
        self.rrf_k = rrf_k

        # BM25初始化
        self.bm25 = BM25Retriever.from_documents(children)
        self.bm25.k = 5 # 默认值，查询时可覆盖

    def vector_search(self, query: str, k: int) -> list[Document]:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def bm25_search(self, query: str, k: int) -> list[Document]:
        self.bm25.k = k
        return self.bm25.invoke(query)
    
    def _doc_key(self, doc: Document) -> str:
        md = doc.metadata or {}
        # 优先 child_id，避免不同对象实例导致去重失败
        return (
            str(md.get("child_id"))
            or f'{md.get("source", "")}::{md.get("chunk_index", "")}'
            or doc.page_content[:100]
        )

    def rrf_fuse(self, vector_docs: list[Document], bm25_docs: list[Document]) -> list[Document]:
        """
        Fuse vector and BM25 results with Reciprocal Rank Fusion (RRF).

        Process:
        1) Iterate each retrieval list and assign rank-based contribution.
        2) Aggregate contributions by a stable document key.
        3) Sort documents by fused score in descending order.
        4) Write fused score back into metadata for debugging/observability.
        """
        # key -> fused RRF score
        scores: dict[str, float] = {}
        # key -> representative document object
        doc_by_key: dict[str, Document] = {}

        # Add contribution from vector retrieval rankings.
        for rank, doc in enumerate(vector_docs):
            key = self._doc_key(doc)
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # Add contribution from BM25 retrieval rankings.
        for rank, doc in enumerate(bm25_docs):
            key = self._doc_key(doc)
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # Rank by fused score, higher score first.
        ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        fused = [doc_by_key[k] for k in ranked_keys]

        # Persist fused score for downstream logging/debug.
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

    def hybrid_search(self, query: str, retrieval_k: int, top_k: int) -> RetrievalResult:
        vector_docs = self.vector_search(query, retrieval_k)
        bm25_docs = self.bm25_search(query, retrieval_k)
        fused_children = self.rrf_fuse(vector_docs, bm25_docs)
        parents = self.child_to_parent(fused_children, top_k)
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
            },
        )