from types import SimpleNamespace


def test_build_domain_agents_from_registry(monkeypatch):
    import agents.domain_agents as domain_agents

    created = []
    bound_clients = []

    monkeypatch.setattr(
        domain_agents,
        "build_chat_model",
        lambda settings, override: object(),
    )
    monkeypatch.setattr(
        domain_agents.registry,
        "to_react_tools",
        lambda names: [f"tool:{name}" for name in names],
    )

    def fake_create_react_agent(*, model, tools, prompt):
        created.append({"tools": tools, "prompt": prompt})
        return created[-1]

    monkeypatch.setattr(domain_agents, "create_react_agent", fake_create_react_agent)
    monkeypatch.setattr(domain_agents, "build_qwen_vl_client", lambda settings: object())
    monkeypatch.setattr(domain_agents, "set_vlm_client", lambda client: bound_clients.append(client))

    settings = SimpleNamespace(image_analysis_enabled=False)
    agents = domain_agents.build_domain_agents(settings)

    assert {"consultation", "diagnostic", "research", "image_analysis"} <= set(agents)
    assert len(created) == 4
    assert all(item["tools"] for item in created)
    assert bound_clients == [None]
