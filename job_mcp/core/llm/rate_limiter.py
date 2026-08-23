"""Token bucket rate limiter for LLM gateway requests."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Async token bucket rate limiter with smooth refill based on elapsed time.

    Enforces requests-per-minute (RPM) limits asynchronously using asyncio.Lock
    to prevent 429 quota exhaustion on free-tier LLM providers.
    """

    def __init__(self, rpm: int = 15, capacity: Optional[float] = None) -> None:
        """Initialize the rate limiter.

        Args:
            rpm: Allowed requests per minute (e.g. 15 for Gemini free-tier).
            capacity: Maximum bucket capacity. Defaults to rpm.
        """
        self._rpm = max(0, rpm)
        self._capacity = float(capacity) if capacity is not None else float(self._rpm)
        self._fill_rate = (self._rpm / 60.0) if self._rpm > 0 else float("inf")
        self._tokens = float(self._capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def rpm(self) -> int:
        """Return the current RPM setting."""
        return self._rpm

    @property
    def capacity(self) -> float:
        """Return the maximum token capacity."""
        return self._capacity

    @property
    def fill_rate(self) -> float:
        """Return the token fill rate per second."""
        return self._fill_rate

    def _refill(self) -> None:
        """Refill tokens based on elapsed monotonic time."""
        if self._rpm <= 0:
            self._tokens = self._capacity
            return

        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._fill_rate)
            self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Get currently available tokens after smooth refill."""
        self._refill()
        return self._tokens

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Synchronously try to acquire token without waiting.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired, False otherwise.
        """
        self._refill()
        if self._rpm <= 0:
            return True
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    async def wait_for_token(self, tokens: float = 1.0) -> None:
        """Asynchronously wait for token availability and consume tokens.

        Args:
            tokens: Number of tokens to acquire (default 1.0).
        """
        if self._rpm <= 0:
            return

        async with self._lock:
            self._refill()
            while self._tokens < tokens:
                needed = tokens - self._tokens
                wait_time = needed / self._fill_rate
                logger.debug(
                    "Rate limit reached: waiting %.3fs for %.2f tokens (rpm=%d)",
                    wait_time,
                    needed,
                    self._rpm,
                )
                await asyncio.sleep(wait_time)
                self._refill()

            self._tokens -= tokens
