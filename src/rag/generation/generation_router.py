from __future__ import annotations

from langchain_core.documents import Document


class GenerationRouter:
    def __init__(self, llm=None, max_context_chars: int = 6000) -> None:
        self.llm = llm
        self.max_context_chars = max_context_chars

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

    def is_insufficient_answer(self, answer: str) -> bool:
        text = (answer or "").lower()
        patterns = [
            "信息不足",
            "资料不足",
            "未检索到",
            "无法确定",
            "无法回答",
            "insufficient information",
            "not enough information",
        ]
        return any(p in text for p in patterns)

    def _build_context(self, parents: list[Document]) -> str:
        blocks = []
        total = 0

        for i, p in enumerate(parents, start=1):
            title = str(p.metadata.get("title", "未知标题"))
            source = str(p.metadata.get("source", ""))
            content = p.page_content.strip()

            images = p.metadata.get("images") or []
            image_lines = []
            for img in images:
                cap = (img.get("caption") or "").strip()
                path = (img.get("image_path") or "").strip()
                if cap or path:
                    image_lines.append(f"- 图片：{cap} {path}".strip())

            block = f"[{i}] 标题：{title}\n来源：{source}\n内容：{content}"
            if image_lines:
                block += "\n相关图片：\n" + "\n".join(image_lines)

            # if total + len(block) > self.max_context_chars:
            #     break
            remaining = self.max_context_chars - total
            if remaining <= 0:
                break

            if len(block) > remaining:
                block = block[:remaining].rstrip() + "\n...（上下文已截断）"
            
            blocks.append(block)
            total += len(block)

        return "\n\n".join(blocks)

    def _build_llm_answer(self, query: str, route: str, parents: list[Document]) -> str:
        context = self._build_context(parents)
        prompt = f"""你是严谨的医学知识库回答生成器。
            要求：
            1. 只能依据【检索上下文】回答，不要编造未出现的事实。
            2. 如果上下文不足以回答，明确说“信息不足”，并说明缺少什么。
            3. 医学问题不能给出确诊或处方，只能做知识解释和就医建议。
            4. 尽量引用上下文编号，例如 [1]、[2]。
            5. 使用中文回答。
            问题：
            {query}
            检索上下文：
            {context}
        """
        resp = self.llm.invoke(prompt)
        text = getattr(resp, "content", "")
        return text.strip() or "信息不足：生成器未返回有效回答。"      

    def build_answer(self, query: str, route: str, parents: list[Document]) -> str:
        if not parents:
            return "未检索到相关内容，请尝试换个问法。"

        if self.llm is None:
            return "信息不足：生成模型未初始化，暂时无法基于检索内容作答，建议改用联网搜索。"

        body = self._build_llm_answer(query, route, parents)
        body = self._append_references(body, parents, max_refs=5)
        body = self._append_image_references(body, parents, max_imgs=5)

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

