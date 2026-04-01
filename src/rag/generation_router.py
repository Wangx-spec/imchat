from __future__ import annotations

from langchain_core.documents import Document

class GenerationRouter:

    def route_query(self, query: str) -> str:
        """
        Route query into one of: list / detail / general.
        """
        q = query.strip().lower()

        general_keywords = ["什么是", "是什么", "原理", "概念", "区别", "作用"]
        list_keywords = ["推荐", "有哪些", "来几道", "列出", "给我几个", "有什么菜"]
        detail_keywords = ["怎么做", "做法", "步骤", "需要什么", "如何做"]

        if any(k in q for k in general_keywords):
            return "general"
        if any(k in q for k in list_keywords):
            return "list"
        if any(k in q for k in detail_keywords):
            return "detail"

        return "general"

    def rewrite_query(self, query: str, route:str) -> str:
        """
        Lightweight rewrite:
        - list/detail: keep original query
        - general: normalize whitespace/punctuation
        """
        if route in {"list", "detail"}:
            return query.strip()
        
        q = " ".join(query.split()).strip()
        return q

    def build_answer(self, query: str, route: str, parents: list[Document]) -> str:
        """
        Build final answer text from routed strategy.
        """
        if not parents:
            return "未检索到相关内容，请尝试换个问法。"
        if route == "list":
            return self._build_list_answer(query, parents)
        if route == "detail":
            return self._build_detail_answer(query, parents)
        return self._build_general_answer(query, parents)

    def _build_list_answer(self, query: str, parents: list[Document]) -> str:
        """
        Return deduplicated titles in list format.
        """
        titles: list[str] = []
        seen: set[str] = set()

        for p in parents:
            title = str(p.metadata.get("title", "")).strip() or "未知标题"
            if title not in seen:
                seen.add(title)
                titles.append(title)
        
        lines = [f"基于检索结果，为你推荐以下内容（问题：{query}）："]
        for i, t in enumerate(titles, start=1):
            lines.append(f"{i}. {t}")
        return "\n".join(lines)
    
    def _build_detail_answer(self, query: str, parents: list[Document]) -> str:
        """
        Use top-1 parent as primary detailed context.
        """
        p0 = parents[0]
        title = str(p0.metadata.get("title", "未知标题"))
        source = str(p0.metadata.get("source", ""))
        content = p0.page_content.strip()

        # keep output concise for MVP
        preview = content[:1000]
        return (
            f"问题：{query}\n\n"
            f"【命中内容】{title}\n"
            f"【来源】{source}\n\n"
            f"{preview}"
        )
    
    def _build_general_answer(self, query: str, parents: list[Document]) -> str:
        """
        Summarize top parents with short snippets.
        """
        lines = [f"问题：{query}", "基于检索到的相关文档，给出如下信息："]

        for i, p in enumerate(parents[:3], start=1):
            title = str(p.metadata.get("title", "未知标题"))
            source = str(p.metadata.get("source", ""))
            snippet = p.page_content.strip().replace("\n", " ")[:180]
            lines.append(f"{i}) {title} | {source}")
            lines.append(f"   摘要：{snippet}")
            
        return "\n".join(lines)