from types import SimpleNamespace

from rag.retrieval.index_store import QdrantIndexStore, LocalFAISSIndexStore, build_index_store


def _cfg(provider: str):
    return SimpleNamespace(
        vector_db_provider=provider,
        index_dir="data/rag_index",
        embedding_api_key="k",
        embedding_model="text-embedding-v4",
        embedding_base_url="https://example.com/v1",
        embedding_dimensions=1024,
        source_dirs=[],
        chunk_size=800,
        chunk_overlap=120,
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        qdrant_collection="rag_documents",
    )


def test_index_store_factory_defaults_to_faiss():
    store = build_index_store(_cfg("faiss"))
    assert isinstance(store, LocalFAISSIndexStore)


def test_index_store_factory_supports_qdrant():
    store = build_index_store(_cfg("qdrant"))
    assert isinstance(store, QdrantIndexStore)
from types import SimpleNamespace

from rag.retrieval.index_store import QdrantIndexStore, LocalFAISSIndexStore, build_index_store


def _cfg(provider: str):
    return SimpleNamespace(
        vector_db_provider=provider,
        index_dir="data/rag_index",
        embedding_api_key="k",
        embedding_model="text-embedding-v4",
        embedding_base_url="https://example.com/v1",
        embedding_dimensions=1024,
        source_dirs=[],
        chunk_size=800,
        chunk_overlap=120,
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        qdrant_collection="rag_documents",
    )


def test_index_store_factory_defaults_to_faiss():
    store = build_index_store(_cfg("faiss"))
    assert isinstance(store, LocalFAISSIndexStore)


def test_index_store_factory_supports_qdrant():
    store = build_index_store(_cfg("qdrant"))
    assert isinstance(store, QdrantIndexStore)
