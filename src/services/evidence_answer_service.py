from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from actions.knowledge_base_tools import get_rag_service
from actions.web_search_tools import web_search

logger = logging.getLogger(__name__)

NO_SOURCE_ANSWER = (
    "当前没有接入联网搜索和可用知识库，无法基于可靠来源回答该问题。"
    "请先启用本地医学知识库或联网搜索后再试。"
)

IDENTITY_ANSWER = (
    "我是一个医学智能助手，可以帮助你整理症状信息、提示可能风险、"
    "结合本地医学知识库或联网搜索提供带来源的参考信息。\n\n"
    "我不能替代医生诊断，也不能直接开具处方。"
    "如果出现持续剧烈头痛、胸痛、呼吸困难、意识异常、持续呕吐等紧急情况，"
    "请及时就医或联系急救服务。"
)

_IDENTITY_PAT = re.compile(
    r"^(你是谁|你是啥|你是什么|介绍一下你自己|介绍下你自己|你能做什么|你可以做什么|你的功能|你有什么功能)[？?!.。！\s]*$"
)


@dataclass
class EvidenceAnswer:
    handled: bool
    answer: str = ""
    source: str = "none"
    citations: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def answer_with_evidence(query: str, settings: Any) -> EvidenceAnswer:
    """Deterministically answer text queries with KB first, web fallback, or refusal."""
    q = (query or "").strip()
    if not q:
        return EvidenceAnswer(handled=False, debug={"reason": "empty_query"})

    if _is_identity_query(q):
        return EvidenceAnswer(
            handled=True,
            answer=IDENTITY_ANSWER,
            source="assistant_identity",
            debug={"intent": "identity"},
        )

    kb_result = _try_kb_answer(q)
    web_enabled = bool(getattr(settings, "tavily_enabled", False))

    if kb_result is not None:
        answer, citations, insufficient = kb_result
        if citations and (not insufficient or not web_enabled):
            return EvidenceAnswer(
                handled=True,
                answer=_append_citations(answer, "知识库来源", citations),
                source="knowledge_base",
                citations=citations,
                debug={"kb_insufficient": insufficient},
            )
        if not web_enabled:
            return EvidenceAnswer(
                handled=True,
                answer=_append_citations(answer, "知识库来源", citations),
                source="knowledge_base",
                citations=citations,
                debug={"kb_insufficient": insufficient, "web_enabled": False},
            )

    if web_enabled:
        web_result = _try_web_answer(q)
        if web_result is not None:
            answer, citations = web_result
            if citations:
                return EvidenceAnswer(
                    handled=True,
                    answer=_append_citations(answer, "联网来源", citations),
                    source="web_search",
                    citations=citations,
                    debug={"kb_available": kb_result is not None},
                )

    if kb_result is not None:
        answer, citations, insufficient = kb_result
        return EvidenceAnswer(
            handled=True,
            answer=_append_citations(answer, "知识库来源", citations),
            source="knowledge_base",
            citations=citations,
            debug={"kb_insufficient": insufficient, "web_fallback_failed": web_enabled},
        )

    return EvidenceAnswer(
        handled=True,
        answer=NO_SOURCE_ANSWER,
        source="none",
        debug={"web_enabled": web_enabled, "kb_available": False},
    )


def _is_identity_query(query: str) -> bool:
    normalized = re.sub(r"\s+", "", query.strip())
    return bool(_IDENTITY_PAT.match(normalized))


def _try_kb_answer(query: str) -> tuple[str, list[dict[str, Any]], bool] | None:
    service = get_rag_service()
    if service is None or not bool(getattr(service, "ready", False)):
        return None
    try:
        result = service.answer(query)
    except Exception as exc:
        logger.warning("[EVIDENCE_KB_FAILED] query=%r error=%s", query, exc)
        return None

    answer = str(getattr(result, "answer", "") or "").strip()
    sources = [str(s).strip() for s in (getattr(result, "sources", []) or []) if str(s).strip()]
    citations = [{"id": i + 1, "source": src} for i, src in enumerate(sources)]
    insufficient = bool(getattr(result, "insufficient_info", False))
    if not answer:
        return None
    return answer, citations, insufficient


def _try_web_answer(query: str) -> tuple[str, list[dict[str, Any]]] | None:
    try:
        raw = web_search.invoke({"query": query})
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        logger.warning("[EVIDENCE_WEB_FAILED] query=%r error=%s", query, exc)
        return None

    if not isinstance(payload, dict) or not payload.get("ok"):
        return None
    answer = str(payload.get("answer") or "").strip()
    rows = payload.get("results") if isinstance(payload.get("results"), list) else []
    citations: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip() or "网页结果"
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        if not url and not content:
            continue
        citations.append(
            {
                "id": len(citations) + 1,
                "title": title,
                "url": url,
                "content": content[:200],
            }
        )
    if not answer:
        return None
    return answer, citations


def _append_citations(answer: str, heading: str, citations: list[dict[str, Any]]) -> str:
    text = _normalize_answer_sources(answer, citations)
    if not citations:
        return text
    lines = [text, "", f"{heading}："]
    for item in citations:
        idx = item.get("id") or len(lines)
        title = str(item.get("title") or item.get("source") or "来源").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        line = f"[{idx}] {title}"
        if url:
            line += f" {url}"
        if content:
            line += f" - {content}"
        lines.append(line)
    return "\n".join(lines).strip()


def _normalize_answer_sources(answer: str, citations: list[dict[str, Any]]) -> str:
    text = (answer or "").strip()
    if not text:
        return text

    # RAGService.build_answer already appends "参考文档"; evidence answers use one
    # unified source block below, so strip the internal reference block first.
    text = re.sub(r"\n{0,2}参考文档：[\s\S]*$", "", text).strip()
    text = re.sub(r"\n{0,2}知识库来源：[\s\S]*$", "", text).strip()
    text = re.sub(r"\n{0,2}联网来源：[\s\S]*$", "", text).strip()

    # If source deduplication leaves only one document, normalize any generated
    # context citation like [2] back to the only visible source id [1].
    if len(citations) == 1:
        text = re.sub(r"\[(\d+)\]", "[1]", text)

    return text
