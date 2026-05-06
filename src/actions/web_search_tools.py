import json
import logging
from typing import Any

from config.settings import load_settings

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger("chat.tools")


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

    try:
        search_tool = TavilySearchResults(max_results=5)
        docs = search_tool.invoke(q)
        if isinstance(docs, str):
            logger.warning("[TOOL_RESULT] name=web_search unexpected_str_payload len=%d", len(docs))
            return _build_web_payload(
                query=q,
                ok=False,
                answer="",
                error="unexpected_response_format",
            )
        if not isinstance(docs, list):
            return _build_web_payload(
                query=q,
                ok=False,
                answer="",
                error="unexpected_response_type",
            )

        lines: list[str] = []
        normalized_results: list[dict[str, Any]] = []
        for item in docs:
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

        answer = "\n".join(lines) if lines else "未检索到相关网页结果。"
        logger.info("[TOOL_RESULT] name=web_search ok result_count=%d", len(normalized_results))
        return _build_web_payload(query=q, ok=True, answer=answer, results=normalized_results)
    except Exception as exc:
        logger.exception("[TOOL_RESULT] name=web_search error=%s", exc)
        return _build_web_payload(query=q, ok=False, answer="", error=f"tool_exception:{exc}")