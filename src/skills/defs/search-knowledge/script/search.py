import json

from actions.knowledge_base_tools import get_rag_service


def search_knowledge(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return json.dumps({"ok": False, "error": "empty_query"}, ensure_ascii=False)

    service = get_rag_service()
    if service is None:
        return json.dumps({"ok": False, "error": "service_not_initialized"}, ensure_ascii=False)

    result = service.answer(q)
    return json.dumps(
        {
            "ok": True,
            "query": q,
            "answer": getattr(result, "answer", ""),
            "sources": getattr(result, "sources", []),
            "debug": getattr(result, "debug", {}),
            "insufficient_info": getattr(result, "insufficient_info", False),
        },
        ensure_ascii=False,
    )
