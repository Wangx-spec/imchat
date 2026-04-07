from __future__ import annotations

from langchain_core.documents import Document


class GenerationRouter:
    def route_query(self, query: str) -> str:
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

    def rewrite_query(self, query: str, route: str) -> str:
        if route in {"list", "detail"}:
            return query.strip()
        return " ".join(query.split()).strip()

    def build_answer(self, query: str, route: str, parents: list[Document]) -> str:
        if not parents:
            return "未检索到相关内容，请尝试换个问法。"

        if route == "list":
            body = self._build_list_answer(query, parents)
        elif route == "detail":
            body = self._build_detail_answer(query, parents)
        else:
            body = self._build_general_answer(query, parents)

        # 统一在回答尾部附上引用文档
        return self._append_references(body, parents, max_refs=5)

    def _collect_references(self, parents: list[Document], max_refs: int = 5) -> list[tuple[str, str]]:
        refs: list[tuple[str, str]] = []
        seen: set[str] = set()

        for p in parents:
            title = str(p.metadata.get("title", "")).strip() or "未知标题"
            source = str(p.metadata.get("source", "")).strip()
            key = source or title
            if not key or key in seen:
                continue
            seen.add(key)
            refs.append((title, source))
            if len(refs) >= max_refs:
                break
        return refs

    def _append_references(self, answer: str, parents: list[Document], max_refs: int = 5) -> str:
        refs = self._collect_references(parents, max_refs=max_refs)
        if not refs:
            return answer

        lines = [answer, "", "参考文档："]
        for i, (title, source) in enumerate(refs, start=1):
            if source:
                lines.append(f"[{i}] {title} | {source}")
            else:
                lines.append(f"[{i}] {title}")
        return "\n".join(lines)

    def _build_list_answer(self, query: str, parents: list[Document]) -> str:
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
        p0 = parents[0]
        title = str(p0.metadata.get("title", "未知标题"))
        content = p0.page_content.strip()
        preview = content[:1000]

        return (
            f"问题：{query}\n\n"
            f"【命中内容】{title}\n\n"
            f"{preview}"
        )

    def _build_general_answer(self, query: str, parents: list[Document]) -> str:
        lines = [f"问题：{query}", "基于检索到的相关文档，给出如下信息："]
        for i, p in enumerate(parents[:3], start=1):
            title = str(p.metadata.get("title", "未知标题"))
            snippet = p.page_content.strip().replace("\n", " ")[:180]
            lines.append(f"{i}) {title}")
            lines.append(f"   摘要：{snippet}")
        return "\n".join(lines)