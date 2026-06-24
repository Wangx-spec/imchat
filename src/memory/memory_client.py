from __future__ import annotations

import logging
from typing import Any, Protocol

from memory.vector_memory_client import VectorMemoryClient

logger = logging.getLogger("memory")


class MemoryClientProtocol(Protocol):
    def recall(self, user_id: str, query: str, limit: int = 5) -> list[str]: ...

    def remember(
        self,
        user_id: str,
        messages: list[dict],
        session_id: str | None = None,
    ) -> None: ...


def build_memory_client(settings: Any) -> MemoryClientProtocol | None:
    backend = str(getattr(settings, "memory_backend", "off") or "off").strip().lower()
    if backend in {"", "off", "none", "disabled"}:
        return None
    if backend != "vector":
        logger.warning("[MEMORY] unsupported backend=%s, disabled", backend)
        return None
    if not bool(getattr(settings, "vector_memory_enabled", True)):
        return None
    try:
        return VectorMemoryClient(settings)
    except Exception as exc:
        logger.warning("[MEMORY] vector client build failed: %s", exc)
        return None
