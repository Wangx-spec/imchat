import json
import logging
from typing import Any

from config.settings import load_settings
from rag.core.circuit_breaker import CircuitBreaker

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

logger = logging.getLogger("chat.tools")

_web_search_breaker: CircuitBreaker | None = None


def _get_web_search_breaker(settings: Any) -> CircuitBreaker | None:
    """惰性初始化 web_search 的熔断器（按进程单例）。"""
    global _web_search_breaker
    if not getattr(settings, "breaker_enabled", True):
        return None
    if _web_search_breaker is None:
        _web_search_breaker = CircuitBreaker(
            fail_threshold=getattr(settings, "breaker_fail_threshold", 3),
            recovery_s=getattr(settings, "breaker_recovery_s", 30.0),
        )
    return _web_search_breaker


def _extract_result_items(docs: Any) -> list[dict[str, Any]] | None:
    """兼容 TavilySearch 的多种返回结构：

    - dict，含 "results": [...]
    - dict，含嵌套的 "artifact" / "data" / "output" / "response"
    - list[dict]
    - JSON string
    其他结构返回 None 表示无法解析。
    """
    if isinstance(docs, str):
        try:
            docs = json.loads(docs)
        except json.JSONDecodeError:
            return None

    if isinstance(docs, dict):
        results = docs.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
        if isinstance(results, dict):
            nested = _extract_result_items(results)
            if nested is not None:
                return nested
        for key in ("artifact", "data", "output", "response", "raw"):
            nested_value = docs.get(key)
            if nested_value is None:
                continue
            nested = _extract_result_items(nested_value)
            if nested is not None:
                return nested
        for value in docs.values():
            if isinstance(value, list) and any(
                isinstance(item, dict) and {"title", "url", "content"} & set(item)
                for item in value
            ):
                return [item for item in value if isinstance(item, dict)]
        return None
    if isinstance(docs, list):
        return [item for item in docs if isinstance(item, dict)]
    return None


def _extract_answer_text(docs: Any) -> str:
    if isinstance(docs, str):
        try:
            docs = json.loads(docs)
        except json.JSONDecodeError:
            return docs.strip()
    if not isinstance(docs, dict):
        return ""
    for key in ("answer", "summary"):
        value = docs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("artifact", "data", "output", "response", "raw"):
        value = docs.get(key)
        if value is None:
            continue
        nested = _extract_answer_text(value)
        if nested:
            return nested
    return ""


def _build_web_payload(
    query: str,
    ok: bool,
    answer: str,
    results: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> str:
    payload = {
        "type": "web_search_result",
        "ok": ok,
        "query": query,
        "answer": answer,
        "results": results or [],
        "error": error,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def web_search(query: str) -> str:
    """Search current medical/web information using Tavily."""
    q = (query or "").strip()
    logger.info("[TOOL_CALL] name=web_search query=%s", q)

    if not q:
        return _build_web_payload(query=q, ok=False, answer="", error="empty_query")

    settings = load_settings()
    if not settings.tavily_enabled:
        logger.info("[TOOL_CALL] name=web_search skipped reason=tavily_disabled")
        return _build_web_payload(
            query=q,
            ok=False,
            answer="",
            error="tavily_disabled",
        )

    breaker = _get_web_search_breaker(settings)
    if breaker is not None and not breaker.allow():
        logger.info("[TOOL_CALL] name=web_search skipped reason=breaker_open")
        return _build_web_payload(query=q, ok=False, answer="", error="breaker_open")

    try:
        search_kwargs: dict[str, Any] = {"max_results": 5}
        if settings.tavily_api_key:
            search_kwargs["api_key"] = settings.tavily_api_key
        search_tool = TavilySearch(**search_kwargs)
        docs = search_tool.invoke(q)

        items = _extract_result_items(docs)
        if items is None:
            preview = docs if isinstance(docs, str) else list(docs.keys()) if isinstance(docs, dict) else type(docs).__name__
            logger.warning(
                "[TOOL_RESULT] name=web_search unexpected_payload type=%s preview=%s",
                type(docs).__name__,
                preview,
            )
            return _build_web_payload(
                query=q,
                ok=False,
                answer="",
                error=f"unexpected_response_format:{preview}"[:200],
            )

        lines: list[str] = []
        normalized_results: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip()
            score = item.get("score")
            normalized_results.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                    "score": score,
                }
            )
            line = f"- {title}"
            if url:
                line += f" | {url}"
            if content:
                line += f" | {content[:200]}"
            lines.append(line)

        tavily_answer = _extract_answer_text(docs)
        answer_parts = []
        if tavily_answer:
            answer_parts.append(tavily_answer)
        if lines:
            answer_parts.append("\n".join(lines))
        answer = "\n\n".join(answer_parts) if answer_parts else "未检索到相关网页结果。"
        logger.info("[TOOL_RESULT] name=web_search ok result_count=%d", len(normalized_results))
        if breaker is not None:
            breaker.record_success()
        return _build_web_payload(query=q, ok=True, answer=answer, results=normalized_results)
    except Exception as exc:
        logger.exception("[TOOL_RESULT] name=web_search error=%s", exc)
        if breaker is not None:
            breaker.record_failure()
        return _build_web_payload(query=q, ok=False, answer="", error=f"tool_exception:{exc}")