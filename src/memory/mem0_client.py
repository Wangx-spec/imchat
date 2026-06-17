from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger("memory.mem0")


@dataclass(frozen=True)
class Mem0Config:
    api_key: str
    user_scope: str = "session"


class Mem0Client:
    def __init__(self, cfg: Mem0Config) -> None:
        self.cfg = cfg
        self._client = self._build_client(cfg.api_key)

    @staticmethod
    def _build_client(api_key: str) -> Any:
        try:
            from mem0 import MemoryClient
        except Exception as exc:
            raise RuntimeError(f"mem0_import_failed: {exc}") from exc
        try:
            return MemoryClient(api_key=api_key)
        except Exception as exc:
            raise RuntimeError(f"mem0_init_failed: {exc}") from exc

    def recall(self, user_id: str, query: str, limit: int = 5) -> list[str]:
        uid = (user_id or "").strip()
        q = (query or "").strip()
        if not uid or not q:
            return []
        if limit < 1:
            limit = 1
        try:
            if hasattr(self._client, "search"):
                result = self._client.search(query=q, user_id=uid, limit=limit)
            elif hasattr(self._client, "get_all"):
                result = self._client.get_all(user_id=uid)
            else:
                return []
            return _extract_memory_texts(result, limit=limit)
        except Exception as exc:
            logger.warning("[MEM0] recall failed user_id=%s error=%s", uid, exc)
            return []

    def remember(self, user_id: str, messages: list[dict]) -> None:
        uid = (user_id or "").strip()
        if not uid or not messages:
            return
        try:
            if hasattr(self._client, "add"):
                self._client.add(messages=messages, user_id=uid)
                return
            if hasattr(self._client, "store"):
                self._client.store(messages=messages, user_id=uid)
                return
            logger.warning("[MEM0] client has no add/store method")
        except Exception as exc:
            logger.warning("[MEM0] remember failed user_id=%s error=%s", uid, exc)


def build_mem0_client(settings) -> Mem0Client | None:
    if not bool(getattr(settings, "mem0_enabled", False)):
        return None
    api_key = (getattr(settings, "mem0_api_key", "") or "").strip()
    if not api_key:
        logger.warning("[MEM0] enabled but MEM0_API_KEY is missing")
        return None
    scope = str(getattr(settings, "mem0_user_scope", "session") or "session").strip().lower()
    if scope not in {"session", "global"}:
        scope = "session"
    try:
        return Mem0Client(Mem0Config(api_key=api_key, user_scope=scope))
    except Exception as exc:
        logger.warning("[MEM0] build failed: %s", exc)
        return None


def _extract_memory_texts(result: Any, limit: int) -> list[str]:
    rows: list[Any]
    if isinstance(result, list):
        rows = result
    elif isinstance(result, dict):
        maybe = result.get("results") or result.get("data") or []
        rows = maybe if isinstance(maybe, list) else []
    else:
        rows = []

    out: list[str] = []
    for row in rows:
        if isinstance(row, str):
            text = row.strip()
        elif isinstance(row, dict):
            text = str(
                row.get("memory")
                or row.get("text")
                or row.get("content")
                or row.get("value")
                or ""
            ).strip()
        else:
            text = str(row).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out
