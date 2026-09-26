import secrets
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Optional

from fastapi import Header, HTTPException, Request

from src.utils.config import settings


def require_api_token(x_api_key: Optional[str] = Header(default=None)):
    """Requires `X-API-Key: <API_AUTH_TOKEN>` on the credit-spending /chat endpoint.
    Fails closed: if an Anthropic key is configured but no API_AUTH_TOKEN, chat is
    refused rather than left open. (With no Anthropic key, chat can't spend anything
    and the handler returns its own 503.)"""
    expected = settings.API_AUTH_TOKEN
    if not expected:
        if settings.ANTHROPIC_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="Chat is disabled: set API_AUTH_TOKEN in .env to protect the ANTHROPIC_API_KEY.",
            )
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


class SlidingWindowRateLimiter:
    """In-process per-key limiter (N calls per 60s). Good enough for a single API
    process; multiple workers/replicas would each keep their own counts."""

    def __init__(self, window_seconds: float = 60.0):
        self.window = window_seconds
        self._calls = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        with self._lock:
            calls = self._calls[key]
            while calls and now - calls[0] > self.window:
                calls.popleft()
            if len(calls) >= limit:
                return False
            calls.append(now)
            return True

    def reset(self):
        with self._lock:
            self._calls.clear()


chat_rate_limiter = SlidingWindowRateLimiter()


def chat_rate_limit(request: Request):
    client = request.client.host if request.client else "unknown"
    if not chat_rate_limiter.check(client, settings.CHAT_RATE_LIMIT_PER_MINUTE):
        raise HTTPException(
            status_code=429,
            detail=f"Chat rate limit exceeded ({settings.CHAT_RATE_LIMIT_PER_MINUTE} requests/minute).",
        )
