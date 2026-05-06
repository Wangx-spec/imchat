from config.settings import load_settings
from rag.core.config import build_rag_config, sanitize_rag_config, validate_rag_config
from rag.core.service import RAGService


def _print_answer_result(title: str, result) -> None:
    print(f"\n[{title}]")
    print("route=", result.route)
    print("sources_top3=", result.sources[:3] if result.sources else [])
    print("debug=", result.debug)
    preview = (result.answer or "").replace("\n", " ")[:240]
    print("answer_preview=", preview)


def main() -> None:
    print("[STEP4] load_settings() ...")
    settings = load_settings()
    print("[STEP4] load_settings() done")

    print("[STEP4] build/sanitize/validate config ...")
    cfg = build_rag_config(settings)
    cfg = sanitize_rag_config(cfg)
    validate_rag_config(cfg)
    print("[STEP4] config ready")

    print("[STEP4] RAGService(cfg) ...")
    service = RAGService(cfg)
    print("[STEP4] RAGService(cfg) done")

    print("[STEP4] service.initialize(force_rebuild=False) ...")
    service.initialize(force_rebuild=False)
    print("[STEP4] initialize done, ready=", service.is_ready())
    print("[STEP4] stats=", service.stats())

    # 按文档要求覆盖 3 种路由类型
    q_list = "推荐几道家常菜"
    q_detail = "宫保鸡丁怎么做"
    q_general = "什么是食材相克"

    print(f"[STEP4] answer(list): {q_list!r} ...")
    r1 = service.answer(q_list)
    print("[STEP4] answer(list) done")
    _print_answer_result("LIST", r1)

    print(f"[STEP4] answer(detail): {q_detail!r} ...")
    r2 = service.answer(q_detail)
    print("[STEP4] answer(detail) done")
    _print_answer_result("DETAIL", r2)

    print(f"[STEP4] answer(general): {q_general!r} ...")
    r3 = service.answer(q_general)
    print("[STEP4] answer(general) done")
    _print_answer_result("GENERAL", r3)

    print("\nSTEP4_SMOKE_OK")


if __name__ == "__main__":
    main()
