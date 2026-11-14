"""Thread-safe rolling-window request limiter."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock


class RequestRateLimiter:
    """Limit request starts to a maximum count in any rolling 60-second window."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be greater than zero")
        self.requests_per_minute = requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return

                wait_seconds = 60 - (now - self._timestamps[0])

            time.sleep(max(wait_seconds, 0))
