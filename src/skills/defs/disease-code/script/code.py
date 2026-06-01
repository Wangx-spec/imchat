import json

from actions.knowledge_base_tools import get_rag_service


def disease_code(disease_name: str) -> str:
    name = (disease_name or "").strip()
    if not name:
        return json.dumps({"ok": False, "error": "empty_query"}, ensure_ascii=False)

    service = get_rag_service()
    if service is None:
        return json.dumps({"ok": False, "error": "service_not_initialized"}, ensure_ascii=False)

    result = service.answer(f"{name} ICD-10编码 疾病分类")
    return json.dumps(
        {
            "ok": True,
            "query": name,
            "answer": getattr(result, "answer", ""),
            "sources": getattr(result, "sources", []),
        },
        ensure_ascii=False,
    )
