from config.settings import load_settings
from rag.core.config import build_rag_config, sanitize_rag_config, validate_rag_config
from rag.ingestion.chunking import ParentChildChunker
from rag.ingestion.data_loader import MarkdownDataLoader
from rag.retrieval.index_store import LocalFAISSIndexStore
from rag.retrieval.retriever import HybridRetriever


def main() -> None:
    # 1) Step1: config chain
    print("[STEP1] load_settings() ...")
    settings = load_settings()
    print("[STEP1] load_settings() done")

    print("[STEP1] build_rag_config() ...")
    cfg = build_rag_config(settings)
    print("[STEP1] build_rag_config() done")

    print("[STEP1] sanitize_rag_config() ...")
    cfg = sanitize_rag_config(cfg)
    print("[STEP1] sanitize_rag_config() done")

    print("[STEP1] validate_rag_config() ...")
    validate_rag_config(cfg)
    print("[STEP1] validate_rag_config() done")

    # 2) Step2: load + chunk
    print("[STEP2] MarkdownDataLoader() ...")
    loader = MarkdownDataLoader()
    print("[STEP2] MarkdownDataLoader() done")

    print("[STEP2] load_documents() ...")
    parents = loader.load_documents(cfg.source_dirs)
    print(f"[STEP2] load_documents() done, parents={len(parents)}")
    if not parents:
        raise RuntimeError("No parent documents loaded. Check RAG_SOURCE_DIRS.")

    print("[STEP2] ParentChildChunker() ...")
    chunker = ParentChildChunker(cfg.chunk_size, cfg.chunk_overlap)
    print("[STEP2] ParentChildChunker() done")

    print("[STEP2] build_parent_child() ...")
    children, parent_map, child_parent = chunker.build_parent_child(parents)
    print(
        "[STEP2] build_parent_child() done, "
        f"children={len(children)}, parent_map={len(parent_map)}, child_parent={len(child_parent)}"
    )
    if not children:
        raise RuntimeError("No child chunks created. Check chunking implementation.")

    # 3) Step3: index load/build + retriever
    print("[STEP3] LocalFAISSIndexStore() ...")
    store = LocalFAISSIndexStore(cfg)
    print("[STEP3] LocalFAISSIndexStore() done")

    print("[STEP3] load() ...")
    loaded = store.load(expected_children_count=len(children))
    print(f"[STEP3] load() done, loaded={loaded}")
    if not loaded:
        print("[STEP3] build(children) ...")
        store.build(children)
        print("[STEP3] build(children) done")

        print("[STEP3] save() ...")
        store.save()
        print("[STEP3] save() done")

    print("[STEP3] HybridRetriever() ...")
    retriever = HybridRetriever(
        vectorstore=store.vectorstore,
        children=children,
        parent_map=parent_map,
        child_parent=child_parent,
        rrf_k=cfg.rrf_k,
    )
    print("[STEP3] HybridRetriever() done")

    # Query sample: change freely based on your corpus
    query = "宫保鸡丁怎么做"
    print(f"[STEP3] hybrid_search(query={query!r}) ...")
    result = retriever.hybrid_search(query=query, retrieval_k=cfg.retrieval_k, top_k=cfg.top_k)
    print("[STEP3] hybrid_search() done")

    # 4) Print verification summary
    print("STEP3_SMOKE_OK")
    print(f"index_loaded={loaded}")
    print(f"parents_total={len(parents)} children_total={len(children)}")
    print(f"result_parent_hits={len(result.parents)}")
    print(f"result_sources={result.sources}")
    print(f"result_debug={result.debug}")

    # Optional sample output preview
    if result.parents:
        p0 = result.parents[0]
        print("\n[first_parent]")
        print("title=", p0.metadata.get("title"))
        print("source=", p0.metadata.get("source"))
        preview = p0.page_content[:200].replace("\n", " ")
        print("preview=", preview)


if __name__ == "__main__":
    main()
