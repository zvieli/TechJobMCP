"""Unit and integration tests for TokenBucketRateLimiter and ResilientLLMGateway."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from job_mcp.core.llm.cache import LLMCache
from job_mcp.core.llm.gateway import (
    LLMError,
    LLMProviderError,
    RateLimitOrUnavailableError,
    ResilientLLMGateway,
)
from job_mcp.core.llm.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter(unittest.IsolatedAsyncioTestCase):
    """Test suite for TokenBucketRateLimiter token replenishment and pacing."""

    async def test_initial_state_and_try_acquire(self) -> None:
        """Verify initial capacity and non-blocking try_acquire."""
        limiter = TokenBucketRateLimiter(rpm=60, capacity=2.0)
        self.assertEqual(limiter.rpm, 60)
        self.assertEqual(limiter.capacity, 2.0)
        self.assertAlmostEqual(limiter.fill_rate, 1.0)  # 60 / 60 = 1 token/sec

        self.assertTrue(limiter.try_acquire(1.0))
        self.assertTrue(limiter.try_acquire(1.0))
        self.assertFalse(limiter.try_acquire(1.0))

    async def test_smooth_token_refill(self) -> None:
        """Verify tokens refill smoothly over time."""
        limiter = TokenBucketRateLimiter(rpm=60, capacity=2.0)
        self.assertTrue(limiter.try_acquire(2.0))
        self.assertLess(limiter.available_tokens, 0.2)

        # Sleep 0.2s -> should refill ~0.2 tokens (60 rpm = 1 token/s)
        await asyncio.sleep(0.25)
        self.assertGreaterEqual(limiter.available_tokens, 0.2)

    async def test_wait_for_token(self) -> None:
        """Verify wait_for_token waits appropriately when bucket is empty."""
        # 120 RPM = 2 tokens/sec -> 0.5s per token
        limiter = TokenBucketRateLimiter(rpm=120, capacity=1.0)
        self.assertTrue(limiter.try_acquire(1.0))

        start_time = time.monotonic()
        await limiter.wait_for_token(1.0)
        elapsed = time.monotonic() - start_time

        # Should wait approximately 0.5s
        self.assertGreaterEqual(elapsed, 0.35)

    async def test_unlimited_rpm(self) -> None:
        """Verify rpm <= 0 behaves without rate limiting."""
        limiter = TokenBucketRateLimiter(rpm=0)
        self.assertTrue(limiter.try_acquire(100.0))
        await limiter.wait_for_token(50.0)


class TestResilientLLMGateway(unittest.IsolatedAsyncioTestCase):
    """Test suite for ResilientLLMGateway caching, fallback, and retry behavior."""

    def setUp(self) -> None:
        """Initialize in-memory cache and gateway."""
        self.cache = LLMCache(db_path=":memory:")
        self.rate_limiter = TokenBucketRateLimiter(rpm=600)  # High rate for fast tests
        self.gateway = ResilientLLMGateway(
            cache=self.cache,
            rate_limiter=self.rate_limiter,
            initial_backoff=0.01,  # Fast backoff for testing
            max_retries=2,
            mock_fallback=True,
        )

    async def test_cache_hit_bypasses_provider_and_rate_limiter(self) -> None:
        """Verify cache hit returns immediately without touching rate limiter or providers."""
        question = "What is your primary programming language?"
        expected_answer = "Python and TypeScript."

        # Prime the cache
        self.cache.cache_answer(question, expected_answer)

        # Mock rate_limiter to verify wait_for_token is NOT called
        self.gateway.rate_limiter.wait_for_token = AsyncMock()
        # Mock providers
        self.gateway._call_gemini = AsyncMock()

        result = await self.gateway.ask_question(question, cv_context="Candidate CV")

        self.assertEqual(result, expected_answer)
        self.gateway.rate_limiter.wait_for_token.assert_not_called()
        self.gateway._call_gemini.assert_not_called()

    async def test_cache_miss_acquires_token_and_caches_result(self) -> None:
        """Verify cache miss invokes provider, acquires rate limit token, and stores answer in cache."""
        question = "How many years of experience do you have with Kubernetes?"
        provider_answer = "Over 4 years orchestrating production microservices with Kubernetes."

        self.gateway.gemini_api_key = "fake_gemini_key"
        self.gateway._call_gemini = AsyncMock(return_value=provider_answer)

        self.assertIsNone(self.cache.get_cached_answer(question))

        result = await self.gateway.ask_question(question, cv_context="CV context")

        self.assertEqual(result, provider_answer)
        self.gateway._call_gemini.assert_called_once()
        # Check cache now has it
        self.assertEqual(self.cache.get_cached_answer(question), provider_answer)

    async def test_rate_limit_retry_exponential_backoff(self) -> None:
        """Verify 429 rate limit triggers exponential backoff retry and eventually succeeds."""
        question = "Are you willing to work hybrid in Tel Aviv?"
        success_answer = "Yes, fully open to hybrid work in Tel Aviv."

        attempts = 0

        async def mock_gemini_with_429(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RateLimitOrUnavailableError("HTTP 429 Too Many Requests")
            return success_answer

        self.gateway.gemini_api_key = "fake_gemini_key"
        self.gateway._call_gemini = mock_gemini_with_429

        result = await self.gateway.ask_question(question)

        self.assertEqual(result, success_answer)
        self.assertEqual(attempts, 2)
        self.assertEqual(self.cache.get_cached_answer(question), success_answer)

    async def test_provider_fallback_chain(self) -> None:
        """Verify fallback from Gemini to OpenRouter when Gemini fails."""
        question = "Do you have experience with GraphQL?"
        openrouter_answer = "Yes, built and consumed GraphQL APIs with Apollo and Strawberry."

        self.gateway.gemini_api_key = "fake_gemini_key"
        self.gateway.openrouter_api_key = "fake_openrouter_key"

        # Gemini fails with non-retryable provider error
        self.gateway._call_gemini = AsyncMock(side_effect=LLMProviderError("Gemini Quota Exceeded"))
        # OpenRouter succeeds
        self.gateway._call_openrouter = AsyncMock(return_value=openrouter_answer)

        result = await self.gateway.ask_question(question)

        self.assertEqual(result, openrouter_answer)
        self.gateway._call_gemini.assert_called_once()
        self.gateway._call_openrouter.assert_called_once()

    async def test_mock_fallback_when_offline_or_unconfigured(self) -> None:
        """Verify Mock LLM handles questions when no API keys are provided."""
        gateway = ResilientLLMGateway(
            cache=LLMCache(db_path=":memory:"),
            rate_limiter=TokenBucketRateLimiter(rpm=600),
            gemini_api_key=None,
            openrouter_api_key=None,
            mock_fallback=True,
        )
        # Ollama will fail connection
        gateway._call_ollama = AsyncMock(side_effect=httpx.ConnectError("Ollama offline"))

        exp_answer = await gateway.ask_question("How many years of experience do you have?")
        self.assertIn("7+", exp_answer)

        auth_answer = await gateway.ask_question("Are you legally authorized to work in the US?")
        self.assertIn("authorized", auth_answer.lower())

        sponsor_answer = await gateway.ask_question("Do you require visa sponsorship?")
        self.assertIn("no", sponsor_answer.lower())

        salary_answer = await gateway.ask_question("What is your expected salary?")
        self.assertIn("compensation", salary_answer.lower())

    async def test_error_raised_when_all_fail_and_no_mock(self) -> None:
        """Verify LLMProviderError is raised if all providers fail and mock_fallback is False."""
        gateway = ResilientLLMGateway(
            cache=LLMCache(db_path=":memory:"),
            rate_limiter=TokenBucketRateLimiter(rpm=600),
            gemini_api_key="fake_key",
            mock_fallback=False,
        )
        gateway._call_gemini = AsyncMock(side_effect=LLMProviderError("Fatal error"))
        gateway._call_ollama = AsyncMock(side_effect=LLMProviderError("Ollama failed"))

        with self.assertRaises(LLMProviderError):
            await gateway.ask_question("Any question?")


if __name__ == "__main__":
    unittest.main()
