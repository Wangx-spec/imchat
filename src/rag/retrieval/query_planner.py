from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from prompts.knowledge_base_prompt import build_query_prompt


@dataclass
class QueryPlan:
    original_query: str
    normalized_query: str
    core_terms: list[str]
    query_variants: list[str]
    used_llm: bool
    error: str | None = None
    intent_hint: str = "general"
    entities: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


class LLMQueryPlanner:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen2.5-coder-7b-instruct",
        timeout_ms: int = 2000,
        max_variants: int = 5,
        llm: Any | None = None,
    ) -> None:
        self.max_variants = max_variants
        if llm is not None:
            self.llm = llm
        else:
            if not api_key or not base_url:
                raise ValueError("api_key and base_url are required when llm is not provided")
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0,
                timeout=max(1.0, timeout_ms / 1000.0),
            )

    def _extract_json(self, text: str) -> dict[str, Any]:
        t = (text or "").strip()
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.S).strip()
        return json.loads(t)

    def _to_str_list(self, v: Any, max_n: int) -> list[str]:
        if not isinstance(v, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for x in v:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= max_n:
                break
        return out

    def _normalize_intent(self, v: Any) -> str:
        s = str(v).strip().lower()
        if s in {"detail", "list", "general"}:
            return s
        # 兼容更细粒度标签
        alias = {
            "howto": "detail",
            "procedure": "detail",
            "troubleshoot": "detail",
            "recommend": "list",
            "enumerate": "list",
            "definition": "general",
            "concept": "general",
        }
        return alias.get(s, "general")

    def _to_constraints(self, v: Any) -> dict[str, Any]:
        return v if isinstance(v, dict) else {}

    def _clamp_confidence(self, v: Any) -> float:
        try:
            f = float(v)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, f))

    def _fallback_core_terms(self, query: str) -> list[str]:
        q = (query or "").strip().lower()
        if not q:
            return []
        # 领域无关：保留英文术语、数字、版本号、错误码、中文连续词
        # 示例: gpt-4o, v1.2.3, err_conn_reset, 白菜猪肉炖粉条
        tokens = re.findall(r"[a-z0-9._-]+|[\u4e00-\u9fff]{2,}", q)
        stop = {
            "请问", "帮我", "告诉我", "一下", "这个", "那个", "问题",
            "是什么", "什么", "怎么", "如何", "可以", "关于"
        }
        out: list[str] = []
        seen: set[str] = set()
        for t in tokens:
            t = t.strip()
            if not t or t in stop or t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= 6:
                break
        if out:
            return out
        short = re.sub(r"\s+", " ", q).strip()
        return [short[:16]] if short else []

    def _sanitize_plan_payload(self, query: str, data: dict[str, Any]) -> QueryPlan:
        normalized = str(data.get("normalized_query", "")).strip() or query
        core_terms = self._to_str_list(data.get("core_terms"), 6)
        # 禁止整句原样，且至少 2 个关键词；不满足则走通用兜底
        if len(core_terms) < 2 or (len(core_terms) == 1 and core_terms[0] == normalized):
            core_terms = self._fallback_core_terms(normalized)
        if len(core_terms) < 2:
            core_terms = (core_terms + self._fallback_core_terms(query))[:6]
            
        variants = self._to_str_list(data.get("query_variants"), max(2, self.max_variants))
        if not variants:
            variants = [normalized]
        if normalized not in variants:
            variants.insert(0, normalized)
        # 至少两条，避免退化成单条原问句
        if len(variants) == 1:
            variants.append(f"{normalized} 相关信息")
        variants = variants[: max(2, self.max_variants)]

        # 新增：entities 兜底
        entities = self._to_str_list(data.get("entities"), 12)
        if not entities:
            entities = core_terms[:3]

        return QueryPlan(
            original_query=query,
            normalized_query=normalized,
            core_terms=core_terms[:6],
            query_variants=variants,
            used_llm=True,
            error=None,
            intent_hint=self._normalize_intent(data.get("intent_hint")),
            entities=self._to_str_list(data.get("entities"), 12),
            constraints=self._to_constraints(data.get("constraints")),
            confidence=self._clamp_confidence(data.get("confidence")),
        )

    def plan(self, query: str) -> QueryPlan:
        q = (query or "").strip()
        if not q:
            return QueryPlan("", "", [], [], used_llm=False, error="empty_query")

        prompt = build_query_prompt(q, self.max_variants)
        try:
            out = self.llm.invoke(prompt)
            data = self._extract_json(getattr(out, "content", ""))

            return self._sanitize_plan_payload(q, data)

            # normalized = str(data.get("normalized_query", "")).strip() or q
            # terms = [str(x).strip() for x in (data.get("core_terms") or []) if str(x).strip()]
            # variants = [str(x).strip() for x in (data.get("query_variants") or []) if str(x).strip()]

            # if not variants:
            #     variants = [normalized]
            # if normalized not in variants:
            #     variants.insert(0, normalized)

            # # 去重截断
            # uniq = []
            # seen = set()
            # for v in variants:
            #     if v not in seen:
            #         seen.add(v)
            #         uniq.append(v)
            # variants = uniq[: self.max_variants]

            # return QueryPlan(
            #     original_query=q,
            #     normalized_query=normalized,
            #     core_terms=terms[:6],
            #     query_variants=variants,
            #     used_llm=True,
            #     error=None,
            # )

        except Exception as exc:
            # 最小兜底：不用规则扩展，直接原问句
            return QueryPlan(
                original_query=q,
                normalized_query=q,
                core_terms=[q],
                query_variants=[q],
                used_llm=False,
                error=f"{type(exc).__name__}: {exc}",
            )
