import json
import logging
import re
from typing import Any

from langchain_core.tools import tool

from actions.knowledge_base_tools import get_rag_service

logger = logging.getLogger("chat.tools")

_RECOMMEND_LIMIT = 15
_RECOMMEND_CONFIDENCE_THRESHOLD = 0.45
_RECOMMEND_MIN_CANDIDATES = 2

_EXPAND_RULES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"(难|高难度|[4-5]\s*[星⭐])"), ["starsystem", "4星", "5星"]),
    (re.compile(r"(简单|入门|新手|[1-2]\s*[星⭐])"), ["starsystem", "1星", "2星"]),
    (re.compile(r"(三星|3\s*[星⭐]|中等)"), ["starsystem", "3星"]),
]

def _expand_query(query: str) -> str:
    # 根据用户描述补充少量检索关键词，帮助 RAG 更容易命中星级索引页等列表文档。
    extras: list[str] = []
    for pat, terms in _EXPAND_RULES:
        if pat.search(query):
            extras.extend(terms)
    if not extras:
        return query
    # 用 dict.fromkeys 去重，同时保留追加关键词的原始顺序。
    return query + " " + " ".join(dict.fromkeys(extras))

def _dedup_parents(parents: list[Any], limit: int) -> list[dict]:
    # retrieve() 返回的是 parent 文档列表；这里按 parent_id/source 去重，
    # 并只保留最终回复真正需要展示的最小字段：title 和 source。
    seen: set[str] = set()
    candidates: list[dict] = []
    for p in parents:
        meta = getattr(p, "metadata", {}) or {}
        pid = meta.get("parent_id") or meta.get("source") or ""
        if pid in seen:
            continue
        seen.add(pid)
        title = meta.get("title", "")
        source = meta.get("source", "")
        if not title and not source:
            continue
        candidates.append({"title": title, "source": source})
        if len(candidates) >= limit:
            break
    return candidates

def _build_payload(
    query: str,
    ok: bool,
    candidates: list[dict] | None = None,
    sources_distinct: list[str] | None = None,
    error: str | None = None,
    debug: dict[str, Any] | None = None,
) -> str:
    # 统一工具输出格式，方便上层 Skill Prompt 只围绕 candidates 做安全回答。
    payload = {
        "type": "dish_recommendation_result",
        "ok": ok,
        "query": query,
        "candidates": candidates or [],
        "sources_distinct": sources_distinct or [],
        "limit": _RECOMMEND_LIMIT,
        "error": error,
        "debug": debug or {},
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def recommend_dishes(query: str) -> str:
    """Recommend dish names from the local knowledge base.
    Input: user's original request describing preferences such as difficulty,
    cuisine style, occasion, dietary restrictions, or general 'what to cook'.
    Returns a JSON string with a list of candidate dish titles."""
    # 先做基础清洗，避免把空字符串传给后续检索链路。
    q = (query or "").strip()
    logger.info("[TOOL_CALL] name=recommend_dishes query=%s", q)

    if not q:
        return _build_payload(query=q, ok=False, error="empty_query")

    # 推荐工具与 search_knowledge_base 共用同一个 RAGService 实例。
    svc = get_rag_service()
    if svc is None:
        return _build_payload(query=q, ok=False, error="service_not_initialized")

    try:
        # 这里不改写用户原意，只做轻量扩展，补充少量有助于召回的关键词。
        expanded = _expand_query(q)
        logger.info("[RECOMMEND] expanded_query=%s", expanded)

        # 推荐场景只需要候选 parent，不需要 answer() 的生成式回答。
        ret = svc.retrieve(expanded)

        # 从检索到的 parent 文档中提取可展示的候选菜名，并去掉重复来源。
        candidates = _dedup_parents(ret.parents, _RECOMMEND_LIMIT)
        sources = list(dict.fromkeys(c["source"] for c in candidates if c["source"]))

        debug = dict(getattr(ret, "debug", {}) or {})
        confidence_score = float(debug.get("confidence_score", 0.0) or 0.0)
        is_confident = bool(debug.get("is_confident", False))
        if "is_confident" not in debug:
            is_confident = bool(debug.get("exact_match_hit", False))

        if (
            not is_confident
            or confidence_score < _RECOMMEND_CONFIDENCE_THRESHOLD
            or len(candidates) < _RECOMMEND_MIN_CANDIDATES
        ):
            logger.warning(
                "[RECOMMEND_GUARD] low_confidence query=%r expanded=%r confidence_score=%.3f is_confident=%s parent_hits=%d deduped=%d",
                q,
                expanded,
                confidence_score,
                is_confident,
                len(ret.parents),
                len(candidates),
            )
            return _build_payload(
                query=q,
                ok=False,
                candidates=[],
                sources_distinct=[],
                error="low_confidence_recommendation",
                debug={
                    "expanded_query": expanded,
                    "confidence_score": confidence_score,
                    "is_confident": is_confident,
                    "parent_hits": len(ret.parents),
                    "deduped": len(candidates),
                },
            )

        text = _build_payload(
            query=q,
            ok=True,
            candidates=candidates,
            sources_distinct=sources,
            debug={
                "expanded_query": expanded,
                "confidence_score": confidence_score,
                "is_confident": is_confident,
                "parent_hits": len(ret.parents),
                "deduped": len(candidates),
            },
        )
        logger.info(
            "[TOOL_RESULT] name=recommend_dishes ok candidates=%d",
            len(candidates),
        )
        return text
    except Exception as exc:
        logger.exception("[TOOL_RESULT] name=recommend_dishes error=%s", exc)
        return _build_payload(query=q, ok=False, error=f"tool_exception:{exc}")