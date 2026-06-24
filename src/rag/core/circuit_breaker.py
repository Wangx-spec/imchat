from __future__ import annotations

import threading
import time


class CircuitBreaker:
    """三态熔断器：CLOSED -> OPEN -> HALF_OPEN -> CLOSED。

    - 连续失败达到 fail_threshold 后打开（OPEN），冷却期内 allow() 返回 False，
      调用方据此快速降级，避免每次都等待远程超时。
    - 冷却 recovery_s 秒后进入 HALF_OPEN，放行一次探测；
      探测成功则关闭，失败则重新打开。
    - 线程安全，可用于并发检索路径。
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, fail_threshold: int = 3, recovery_s: float = 30.0) -> None:
        self.fail_threshold = max(1, int(fail_threshold))
        self.recovery_s = max(0.0, float(recovery_s))
        self._state = self.CLOSED
        self._fail_count = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        """是否允许执行真实调用。OPEN 且未到冷却期返回 False。"""
        with self._lock:
            if self._state == self.OPEN:
                if self._opened_at is None:
                    return True
                if (time.monotonic() - self._opened_at) >= self.recovery_s:
                    self._state = self.HALF_OPEN
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._fail_count = 0
            self._state = self.CLOSED
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._fail_count += 1
            if self._fail_count >= self.fail_threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()
