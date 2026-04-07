import logging

from actions.basic_tools import set_rag_service
from config.settings import Settings
from rag.config import build_rag_config, sanitize_rag_config, validate_rag_config
from rag.service import RAGService


logger = logging.getLogger("chat.rag")


def bootstrap_rag(settings: Settings) -> tuple[bool, str]:
    if not settings.rag_enabled:
        return False, "disabled"

    try:
        cfg = sanitize_rag_config(build_rag_config(settings))
        validate_rag_config(cfg)

        service = RAGService(cfg)
        service.initialize(force_rebuild=cfg.rebuild)
        set_rag_service(service)

        st = service.stats()
        # 例如: ok:index_loaded=True,index_rebuilt=False,meta=meta_match
        reason = (
            "ok:"
            f"index_loaded={st.get('index_loaded')},"
            f"index_rebuilt={st.get('index_rebuilt')},"
            f"meta={st.get('index_meta_reason')}"
        )
        return True, reason
    except Exception as exc:
        return False, f"init_error:{exc}"