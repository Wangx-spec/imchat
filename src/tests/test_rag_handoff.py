from rag.core.service import RAGService
from rag.core.types import RAGConfig, RetrievalResult
from rag.generation.generation_router import GenerationRouter


class DetailRouter:
    def route_query(self, query: str) -> str:
        return "detail"

    def rewrite_query(self, query: str, route: str) -> str:
        return query


def test_rag_low_confidence_sets_insufficient_info():
    service = RAGService.__new__(RAGService)
    service.cfg = RAGConfig(enabled=True)
    service.ready = True
    service.router = DetailRouter()

    def fake_retrieve(query: str) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            parents=[],
            sources=["source-a.md"],
            debug={
                "confidence_score": 0.1,
                "is_confident": False,
                "direct_hit_count": 0,
                "top_titles": ["unrelated"],
                "query_normalized": "mri 表现",
            },
        )

    service.retrieve = fake_retrieve

    result = service.answer("脑肿瘤 MRI 有哪些表现？")

    assert result.insufficient_info is True
    assert result.debug["insufficient_info"] is True
    assert result.debug["insufficient_reason"] == "low_confidence"
    assert result.debug["low_confidence_blocked"] is True


def test_generation_router_no_llm_can_trigger_handoff():
    router = GenerationRouter(llm=None)
    answer = router.build_answer(
        "脑肿瘤 MRI 有哪些表现？",
        "detail",
        parents=[type("Doc", (), {"page_content": "context", "metadata": {}})()],
    )

    assert router.is_insufficient_answer(answer) is True
