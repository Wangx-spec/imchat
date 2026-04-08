import json
import re
from typing import Any

from actions.knowledge_base_tools import search_knowledge_base
from prompts.knowledge_base_prompt import build_forced_route_polish_input
from llms.openai_chat import build_openai_chat_model
from config.settings import Settings


_KB_INTENT_PATTERNS = [
    r"怎么做", r"做法", r"步骤", r"教程", r"推荐", r"有哪些", r"配菜", r"几人", r"食谱", r"菜谱"
]


def should_force_kb(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    return any(re.search(p, q) for p in _KB_INTENT_PATTERNS)


def call_kb(query: str) -> dict[str, Any]:
    raw = search_knowledge_base.invoke({"query": query})
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"ok": False, "answer": "", "citations": [], "sources": [], "error": "invalid_kb_payload"}


def build_forced_kb_answer(query: str, kb: dict[str, Any], settings: Settings) -> str:
    ok = bool(kb.get("ok"))
    answer = (kb.get("answer") or "").strip()
    citations = kb.get("citations") or []

    if (not ok) or (not answer):
        return (
            f"知识库没有精确命中“{query}”。\n\n"
            "非知识库兜底建议：请提供更具体关键词（完整菜名/文档标题/别名），我再为你检索。"
        )

    if not settings.rag_force_tool_polish:
        refs = "\n".join([f"[{c.get('id')}] {c.get('source')}" for c in citations[:6]])
        return f"{answer}\n\n参考文档：\n{refs}" if refs else answer

    llm = build_openai_chat_model(settings)
    kb_payload = json.dumps(kb, ensure_ascii=False)
    prompt = build_forced_route_polish_input(query, kb_payload)
    out = llm.invoke(prompt)
    return str(getattr(out, "content", "")).strip() or answer