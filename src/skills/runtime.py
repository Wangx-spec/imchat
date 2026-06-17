# src/skills/runtime.py
from __future__ import annotations

from typing import Any

_vlm_client: Any | None = None


def set_vlm_client(client: Any | None) -> None:
    global _vlm_client
    _vlm_client = client


def get_vlm_client() -> Any | None:
    return _vlm_client