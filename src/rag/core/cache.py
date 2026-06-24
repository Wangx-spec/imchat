from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """线程安全的单进程 TTL + LRU 缓存。

    - 每个条目带独立过期时间（写入时 now + ttl）。
    - 超过 max_size 时按最近最少使用淘汰。
    - get 命中过期条目时惰性删除并视为未命中。
    """

    def __init__(self, ttl_s: float = 300.0, max_size: int = 512) -> None:
        self.ttl_s = max(0.0, float(ttl_s))
        self.max_size = max(1, int(max_size))
        self._store: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, expire_at = item
            if expire_at <= now:
                # 惰性过期清理
                self._store.pop(key, None)
                return None
            # LRU：命中后移到末尾
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        if self.ttl_s <= 0:
            return
        expire_at = time.monotonic() + self.ttl_s
        with self._lock:
            self._store[key] = (value, expire_at)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
