from config.settings import Settings
from rag.core.config import build_rag_config


def test_build_rag_config_maps_qdrant_fields():
    settings = Settings(
        openai_api_key="test",
        vector_db_provider="qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_api_key="secret",
        qdrant_collection="col1",
    )
    cfg = build_rag_config(settings)

    assert cfg.vector_db_provider == "qdrant"
    assert cfg.qdrant_url == "http://localhost:6333"
    assert cfg.qdrant_api_key == "secret"
    assert cfg.qdrant_collection == "col1"
