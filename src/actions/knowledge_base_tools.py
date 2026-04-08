import json
import logging
from typing import Any

from langchain_core.tools import tool


logger = logging.getLogger("chat.tools")
_rag_service: Any = None


def set_rag_service(service: Any) -> None:
    global _rag_service
    _rag_service = service
    logger.info("[RAG_BIND] service=%s", type(service).__name__ if service else "None")


def _build_kb_payload(
    query: str,
    ok: bool,
    answer: str,
    sources: list[str] | None = None,
    route: str | None = None,
    debug: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    srcs = sources or []
    payload = {
        "type": "kb_result",
        "ok": ok,
        "query": query,
        "answer": (answer or "").strip(),
        "sources": srcs,
        "citations": [{"id": i + 1, "source": s} for i, s in enumerate(srcs)],
        "route": route,
        "debug": debug or {},
        "error": error,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def search_knowledge_base(query: str) -> str:
    """Search local knowledge base and return answer with sources."""
    q = (query or "").strip()
    logger.info("[TOOL_CALL] name=search_knowledge_base query=%s", q)

    if not q:
        return _build_kb_payload(query=q, ok=False, answer="", error="empty_query")
    if _rag_service is None:
        return _build_kb_payload(query=q, ok=False, answer="", error="service_not_initialized")

    try:
        result = _rag_service.answer(q)
        dbg = getattr(result, "debug", {}) or {}
        logger.info(
            "[TOOL_KB_DEBUG] query=%r variant_queries=%s query_variants_count=%s direct_hit_titles=%s "
            "exact_match_hit=%s confidence_score=%s confidence_reasons=%s query_plan_used_llm=%s "
            "query_plan_error=%s query_normalized=%s query_core_terms=%s low_confidence_blocked=%s",
            q,
            dbg.get("variant_queries", []),
            len(dbg.get("variant_queries", []) or []),
            dbg.get("direct_hit_titles", []),
            dbg.get("exact_match_hit", False),
            dbg.get("confidence_score", 0.0),
            dbg.get("confidence_reasons", []),
            dbg.get("query_plan_used_llm", False),
            dbg.get("query_plan_error"),
            dbg.get("query_normalized", ""),
            dbg.get("query_core_terms", []),
            dbg.get("low_confidence_blocked", False),
        )
        text = _build_kb_payload(
            query=q,
            ok=True,
            answer=getattr(result, "answer", ""),
            sources=getattr(result, "sources", []),
            route=getattr(result, "route", None),
            debug=getattr(result, "debug", {}),
        )
        logger.info("[TOOL_RESULT] name=search_knowledge_base ok")
        return text
    except Exception as exc:
        logger.exception("[TOOL_RESULT] name=search_knowledge_base error=%s", exc)
        return _build_kb_payload(query=q, ok=False, answer="", error=f"tool_exception:{exc}")

