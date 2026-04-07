from config.settings import load_settings
from rag.config import build_rag_config, sanitize_rag_config, validate_rag_config
from rag.service import RAGService


def run_one(query: str, force_rebuild: bool) -> None:
    settings = load_settings()
    cfg = sanitize_rag_config(build_rag_config(settings))
    validate_rag_config(cfg)

    service = RAGService(cfg)
    service.initialize(force_rebuild=force_rebuild)

    print("\n" + "=" * 80)
    print(f"QUERY={query!r} | force_rebuild={force_rebuild}")
    print("service_stats =", service.stats())

    ret = service.retrieve(query)
    dbg = ret.debug or {}

    print("sources_top5 =")
    for i, s in enumerate(ret.sources[:5], start=1):
        print(f"  {i}. {s}")

    # 关键诊断字段（你在 retriever.py 里加过）
    keys = [
        "vector_hits",
        "bm25_hits",
        "fused_hits",
        "parent_hits",
        "rrf_k",
        "candidate_parent_k",
        "candidate_parent_count_before_trim",
        "top_titles_before_rerank",
        "top_titles",
        "query_tokens",
        "rerank_applied",
        "qwen_rerank_enabled",
        "qwen_rerank_called",
        "qwen_rerank_ok",
        "qwen_rerank_error",
        "qwen_rerank_model",
        "qwen_rerank_scores",
    ]
    print("\ndebug_fields =")
    for k in keys:
        if k in dbg:
            print(f"  {k}: {dbg[k]}")

    # 便于你快速看是否命中目标文档
    joined = "\n".join(ret.sources)
    print("\ncontains_奥利奥冰淇淋 =", ("奥利奥冰淇淋" in joined))


def main() -> None:
    queries = [
        "奥利奥冰淇淋怎么做",
        "奥利奥冰淇淋做法",
        "姜葱捞鸡怎么做",
        "姜炒鸡怎么做",
    ]

    for q in queries:
        run_one(q, force_rebuild=False)

    # 可选：再做一次重建对比
    print("\n" + "#" * 80)
    print("FORCE_REBUILD ONCE FOR A/B CHECK")
    run_one("奥利奥冰淇淋怎么做", force_rebuild=True)


if __name__ == "__main__":
    main()