import logging

from actions.basic_tools import set_rag_service
from config.settings import Settings
from rag.config import build_rag_config, sanitize_rag_config, validate_rag_config
from rag.service import RAGService


logger = logging.getLogger("chat.rag")


def bootstrap_rag(settings: Settings) -> tuple[bool, str]:
    if not settings.rag_enabled:
        return False, "disabled"

    cfg = sanitize_rag_config(build_rag_config(settings))
    validate_rag_config(cfg)

    service = RAGService(cfg)
    service.initialize(force_rebuild=cfg.rebuild)
    set_rag_service(service)

    return True, "ok"