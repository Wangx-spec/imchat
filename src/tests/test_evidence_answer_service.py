from types import SimpleNamespace

from services import evidence_answer_service as svc


class FakeRAG:
    ready = True

    def __init__(self, answer="kb answer", sources=None, insufficient=False):
        self._answer = answer
        self._sources = sources if sources is not None else ["doc.md"]
        self._insufficient = insufficient

    def answer(self, query):
        return SimpleNamespace(
            answer=self._answer,
            sources=self._sources,
            insufficient_info=self._insufficient,
        )


class FakeWebTool:
    def invoke(self, payload):
        return (
            '{"ok": true, "answer": "web answer", "results": ['
            '{"title": "source", "url": "https://example.com", "content": "web content"}'
            "]}"
        )


def test_evidence_answer_prefers_kb(monkeypatch):
    monkeypatch.setattr(svc, "get_rag_service", lambda: FakeRAG())
    result = svc.answer_with_evidence("头痛怎么办", SimpleNamespace(tavily_enabled=True))

    assert result.handled is True
    assert result.source == "knowledge_base"
    assert "kb answer" in result.answer
    assert "知识库来源" in result.answer


def test_evidence_answer_normalizes_kb_source_block(monkeypatch):
    answer = (
        "根据检索上下文，剂量为两剂或三剂 [2]。\n\n"
        "信息来源：[2]\n\n"
        "参考文档：\n"
        "[1] HPV文档 | /tmp/hpv.md"
    )
    monkeypatch.setattr(svc, "get_rag_service", lambda: FakeRAG(answer=answer, sources=["/tmp/hpv.md"]))

    result = svc.answer_with_evidence("HPV疫苗接种的剂量？", SimpleNamespace(tavily_enabled=False))

    assert "剂量为两剂或三剂 [1]" in result.answer
    assert "信息来源：[1]" in result.answer
    assert "参考文档：" not in result.answer
    assert result.answer.count("知识库来源：") == 1
    assert "[1] /tmp/hpv.md" in result.answer


def test_evidence_answer_handles_identity_query_without_sources(monkeypatch):
    monkeypatch.setattr(svc, "get_rag_service", lambda: None)

    result = svc.answer_with_evidence("你是谁", SimpleNamespace(tavily_enabled=False))

    assert result.handled is True
    assert result.source == "assistant_identity"
    assert "医学智能助手" in result.answer
    assert "当前没有接入联网搜索" not in result.answer


def test_evidence_answer_uses_web_when_kb_insufficient(monkeypatch):
    monkeypatch.setattr(svc, "get_rag_service", lambda: FakeRAG(sources=[], insufficient=True))
    monkeypatch.setattr(svc, "web_search", FakeWebTool())

    result = svc.answer_with_evidence("最新指南", SimpleNamespace(tavily_enabled=True))

    assert result.handled is True
    assert result.source == "web_search"
    assert "web answer" in result.answer
    assert "联网来源" in result.answer
    assert "https://example.com" in result.answer


def test_evidence_answer_refuses_without_kb_or_web(monkeypatch):
    monkeypatch.setattr(svc, "get_rag_service", lambda: None)
    result = svc.answer_with_evidence("头痛怎么办", SimpleNamespace(tavily_enabled=False))

    assert result.handled is True
    assert result.source == "none"
    assert "当前没有接入联网搜索和可用知识库" in result.answer
