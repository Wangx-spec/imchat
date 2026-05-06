from config.settings import load_settings
from rag.core.config import build_rag_config, sanitize_rag_config, validate_rag_config
from rag.ingestion.chunking import ParentChildChunker
from rag.ingestion.data_loader import MarkdownDataLoader


def main() -> None:
    # 1) 配置链路
    settings = load_settings()
    cfg = build_rag_config(settings)
    cfg = sanitize_rag_config(cfg)
    validate_rag_config(cfg)

    # 2) 数据加载
    loader = MarkdownDataLoader()
    parents = loader.load_documents(cfg.source_dirs)

    # 3) 分块
    chunker = ParentChildChunker(cfg.chunk_size, cfg.chunk_overlap)
    children, parent_map, child_parent = chunker.build_parent_child(parents)

    # 4) 打印统计
    print("STEP2_SMOKE_OK")
    print(f"parents={len(parents)}")
    print(f"children={len(children)}")
    print(f"parent_map={len(parent_map)}")
    print(f"child_parent={len(child_parent)}")

    # 5) 抽样检查
    if parents:
        p0 = parents[0]
        print("\n[parent sample]")
        print("title=", p0.metadata.get("title"))
        print("source=", p0.metadata.get("source"))
        print("doc_type=", p0.metadata.get("doc_type"))

    if children:
        c0 = children[0]
        print("\n[child sample]")
        print("child_id=", c0.metadata.get("child_id"))
        print("parent_id=", c0.metadata.get("parent_id"))
        print("chunk_index=", c0.metadata.get("chunk_index"))
        print("chunk_size=", c0.metadata.get("chunk_size"))
        print("doc_type=", c0.metadata.get("doc_type"))

        # 映射一致性 quick check
        cid = c0.metadata.get("child_id")
        pid = c0.metadata.get("parent_id")
        mapped_pid = child_parent.get(cid)
        print("\n[mapping check]")
        print("child_parent[cid] == parent_id ?", mapped_pid == pid)


if __name__ == "__main__":
    main()