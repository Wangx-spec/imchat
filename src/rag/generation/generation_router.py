from __future__ import annotations

from langchain_core.documents import Document


class GenerationRouter:

    def _collect_image_refs(self, parents: list[Document], max_imgs: int = 5) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        for p in parents:
            for img in (p.metadata.get("images") or []):
                key = img.get("image_id") or img.get("image_path")
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(img)
                if len(out) >= max_imgs:
                    return out
        return out

    def _append_image_references(self, answer: str, parents: list[Document], max_imgs: int = 5) -> str:
        imgs = self._collect_image_refs(parents, max_imgs=max_imgs)
        if not imgs:
            return answer
        lines = [answer, "", "参考图片："]
        for i, img in enumerate(imgs, start=1):
            cap = (img.get("caption") or "").strip()
            path = (img.get("image_path") or "").strip()
            if cap and path:
                lines.append(f"[{i}] {cap} | {path}")
            elif path:
                lines.append(f"[{i}] {path}")
            elif cap:
                lines.append(f"[{i}] {cap}")
        return "\n".join(lines)

    def route_query(self, query: str) -> str:
            q = query.strip().lower()
            # 医学问题里常见的“列举型知识问答”，应视为 detail/general，
            # 不能因为出现“有哪些”就走旧的推荐列表逻辑。
            medical_detail_keywords = [
                "典型表现", "表现", "征象", "特征", "影像学", "mri", "ct", "dwi", "adc",
                "病理", "分子", "诊断", "诊断标准", "鉴别", "区别", "分级", "分型",
                "机制", "原因", "并发症", "风险因素", "治疗原则", "评估", "指南",
            ]
            medical_list_phrases = [
                "有哪些表现", "有哪些特征", "有哪些征象", "有哪些类型",
                "有哪些诊断标准", "有哪些影像学特点", "有哪些变化",
            ]
            general_keywords = ["什么是", "是什么", "原理", "概念", "区别", "作用"]
            list_keywords = ["推荐", "列出", "给我几个"]
            detail_keywords = ["怎么做", "做法", "步骤", "需要什么", "如何做"]
            # 先识别医学 detail/list 型问题
            if any(p in q for p in medical_list_phrases):
                return "detail"
            if any(k in q for k in medical_detail_keywords):
                return "detail"
            # 再走原有通用规则
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

        body = self._append_references(body, parents, max_refs=5)
        body = self._append_image_references(body, parents, max_imgs=5)
        # 统一在回答尾部附上引用文档
        return body

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
        lines = [f"基于检索结果，整理出以下相关条目（问题：{query}）："]
        seen: set[str] = set()
        idx = 1
        for p in parents:
            title = str(p.metadata.get("title", "")).strip() or "未知标题"
            if title in seen:
                continue
            seen.add(title)
            snippet = p.page_content.strip().replace("\n", " ")[:120]
            lines.append(f"{idx}. {title}")
            if snippet:
                lines.append(f"   摘要：{snippet}")
            idx += 1
            if idx > 5:
                break
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
