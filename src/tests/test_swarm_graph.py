from types import SimpleNamespace

from langchain_core.messages import AIMessage


class FakeLLM:
    def invoke(self, messages):
        return AIMessage(content='{"agent": "consultation", "reasoning": "test", "confidence": 1.0, "complexity": 0.1}')


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def invoke(self, inputs):
        return {"messages": [AIMessage(content=f"{self.name} answer")]}


def test_swarm_graph_compiles_without_external_services(monkeypatch):
    import graphs.swarm_graph as swarm_graph

    monkeypatch.setattr(swarm_graph, "build_openai_chat_model", lambda settings: FakeLLM())
    monkeypatch.setattr(swarm_graph, "build_qwen_vl_client", lambda settings: None)
    monkeypatch.setattr(
        swarm_graph,
        "build_domain_agents",
        lambda settings: {
            "consultation": FakeAgent("consultation"),
            "diagnostic": FakeAgent("diagnostic"),
            "research": FakeAgent("research"),
            "image_analysis": FakeAgent("image_analysis"),
        },
    )

    settings = SimpleNamespace(
        guardrails_enabled=False,
        image_analysis_enabled=False,
        multimodal_enabled=False,
        multimodal_allowed_mime=["image/png", "image/jpeg", "image/webp"],
        multimodal_max_image_bytes=8 * 1024 * 1024,
        multimodal_max_images_per_request=3,
        triage_confidence_threshold=0.75,
        complexity_threshold=0.6,
        swarm_max_workers=3,
    )

    graph = swarm_graph.build_swarm_graph(settings)

    assert hasattr(graph, "invoke")
