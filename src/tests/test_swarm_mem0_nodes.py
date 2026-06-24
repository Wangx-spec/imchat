from types import SimpleNamespace

from langchain_core.messages import AIMessage


class FakeLLM:
    def invoke(self, messages):
        return AIMessage(
            content='{"agent": "consultation", "reasoning": "test", "confidence": 1.0, "complexity": 0.1}'
        )


class CapturingAgent:
    def __init__(self) -> None:
        self.last_inputs = None

    def invoke(self, inputs):
        self.last_inputs = inputs
        return {"messages": [AIMessage(content="consultation answer")]}


class FakeMem0:
    def __init__(self):
        self.remember_calls = []

    def recall(self, user_id: str, query: str, limit: int = 5):
        return ["prefers concise answer"]

    def remember(self, user_id: str, messages: list[dict], session_id: str | None = None):
        self.remember_calls.append((user_id, messages, session_id))


def test_swarm_graph_memory_recall_and_persist(monkeypatch):
    import graphs.swarm_graph as swarm_graph

    agent = CapturingAgent()
    fake_memory = FakeMem0()

    monkeypatch.setattr(swarm_graph, "build_openai_chat_model", lambda settings: FakeLLM())
    monkeypatch.setattr(swarm_graph, "build_qwen_vl_client", lambda settings: None)
    monkeypatch.setattr(
        swarm_graph,
        "build_domain_agents",
        lambda settings: {
            "consultation": agent,
            "diagnostic": agent,
            "research": agent,
            "image_analysis": agent,
        },
    )
    monkeypatch.setattr(swarm_graph, "build_memory_client", lambda settings: fake_memory)
    monkeypatch.setattr(
        swarm_graph,
        "answer_with_evidence",
        lambda query, settings: type(
            "Evidence",
            (),
            {
                "handled": True,
                "answer": "evidence answer",
                "source": "test",
                "citations": [],
                "debug": {},
            },
        )(),
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
        hitl_enabled=False,
        hitl_action="off",
    )

    graph = swarm_graph.build_swarm_graph(settings)
    graph.invoke(
        {"messages": [("user", "请给我建议")]},
        config={"configurable": {"thread_id": "sid-1"}},
    )

    assert agent.last_inputs is None
    assert fake_memory.remember_calls
    assert fake_memory.remember_calls[0][0] == "default-user"
    assert fake_memory.remember_calls[0][2] == "sid-1"
