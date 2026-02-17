from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory fixed window limiter for public read API."""

    def __init__(self, app, per_minute: int = 120):
        super().__init__(app)
        self.per_minute = max(1, per_minute)
        self._state: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api") or path in {"/api/health", "/api/ready"}:
            return await call_next(request)

        now_bucket = int(time.time() // 60)
        client = request.client.host if request.client else "unknown"
        key = f"{client}:{path}"

        with self._lock:
            bucket, count = self._state[key]
            if bucket != now_bucket:
                bucket, count = now_bucket, 0
            count += 1
            self._state[key] = (bucket, count)
            if count > self.per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )

        return await call_next(request)
