from config.settings import load_settings
from rag.config import build_rag_config, sanitize_rag_config, validate_rag_config
from rag.service import RAGService


def dump(service: RAGService, query: str, tag: str) -> None:
    print(f"\n=== {tag} ===")
    route = service.router.route_query(query)
    rewritten = service.router.rewrite_query(query, route)
    ret = service.retrieve(rewritten)

    print("route =", route)
    print("rewritten =", rewritten)
    print("debug =", ret.debug)

    hit_sources = []
    for i, p in enumerate(ret.parents, start=1):
        title = p.metadata.get("title", "")
        source = p.metadata.get("source", "")
        hit_sources.append(source)
        print(f"{i}. {title} | {source}")

    has_kungpao = any("宫保鸡丁" in s for s in hit_sources)
    print("contains_宫保鸡丁_source =", has_kungpao)


def main() -> None:
    query = "宫保鸡丁怎么做"

    settings = load_settings()
    cfg = sanitize_rag_config(build_rag_config(settings))
    validate_rag_config(cfg)

    # A) 复用索引路径
    s1 = RAGService(cfg)
    s1.initialize(force_rebuild=False)
    dump(s1, query, "LOAD_INDEX")

    # B) 强制重建路径
    s2 = RAGService(cfg)
    s2.initialize(force_rebuild=True)
    dump(s2, query, "FORCE_REBUILD")


if __name__ == "__main__":
    main()