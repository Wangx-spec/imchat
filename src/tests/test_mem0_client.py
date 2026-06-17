from memory.mem0_client import build_mem0_client


class FakeRawMem0:
    def __init__(self):
        self.added = []

    def search(self, query, user_id, limit=5):
        return [{"memory": f"{user_id}:{query}:{limit}"}]

    def add(self, messages, user_id):
        self.added.append((user_id, messages))


def test_build_mem0_client_disabled_returns_none():
    class Settings:
        mem0_enabled = False
        mem0_api_key = "x"
        mem0_user_scope = "session"

    assert build_mem0_client(Settings()) is None


def test_mem0_client_recall_and_remember(monkeypatch):
    class Settings:
        mem0_enabled = True
        mem0_api_key = "k"
        mem0_user_scope = "session"

    from memory import mem0_client as mod

    fake = FakeRawMem0()
    monkeypatch.setattr(mod.Mem0Client, "_build_client", staticmethod(lambda key: fake))

    client = build_mem0_client(Settings())
    assert client is not None

    memories = client.recall("u1", "q1", limit=3)
    assert memories == ["u1:q1:3"]

    client.remember("u1", [{"role": "user", "content": "hello"}])
    assert fake.added
