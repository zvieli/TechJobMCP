"""Unit and integration tests for TokenBucketRateLimiter and ResilientLLMGateway."""

from __future__ import annotations

import asyncio
import os
import time
import unittest
from unittest.mock import AsyncMock, patch


import httpx

from job_mcp.core.llm.cache import LLMCache
from job_mcp.models.schemas import CandidateProfile
from job_mcp.core.llm.gateway import (
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
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "OPENROUTER_API_KEY": ""}):
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
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            gateway = ResilientLLMGateway(
                cache=LLMCache(db_path=":memory:"),
                rate_limiter=TokenBucketRateLimiter(rpm=600),
                gemini_api_key="fake_key",
                openrouter_api_key=None,
                mock_fallback=False,
            )
            gateway._call_gemini = AsyncMock(side_effect=LLMProviderError("Fatal error"))
            gateway._call_ollama = AsyncMock(side_effect=LLMProviderError("Ollama failed"))

            with self.assertRaises(LLMProviderError):
                await gateway.ask_question("Any question?")

    def test_env_model_configuration(self) -> None:
        """Verify environment variables correctly override model and base URL settings."""
        with patch.dict(
            os.environ,
            {
                "GEMINI_MODEL": "gemini-flash-lite-latest",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "OPENROUTER_MODEL": "z-ai/glm-5.2:free",
                "OPENROUTER_REASONING_MODEL": "z-ai/glm-5.2:free",
                "OPENROUTER_EXTRACTION_MODEL": "google/gemma-4-26b-a4b-it:free",
            },
        ):
            gw = ResilientLLMGateway(cache=LLMCache(db_path=":memory:"))
            self.assertEqual(gw.gemini_model, "gemini-flash-lite-latest")
            self.assertEqual(gw.openrouter_base_url, "https://openrouter.ai/api/v1")
            self.assertEqual(gw.openrouter_model, "z-ai/glm-5.2:free")
            self.assertEqual(gw.openrouter_reasoning_model, "z-ai/glm-5.2:free")
            self.assertEqual(gw.openrouter_extraction_model, "google/gemma-4-26b-a4b-it:free")

    async def test_generate_personal_note_gemini_success(self) -> None:
        """Verify Gemini successfully generates personal note and caches it."""
        note_content = "I am writing with great enthusiasm for the AI Engineer role at CyberTech..."
        self.gateway.gemini_api_key = "fake_gemini_key"
        self.gateway._call_gemini = AsyncMock(return_value=note_content)

        result = await self.gateway.generate_personal_note(
            job_title="AI Engineer",
            company="CyberTech",
            job_description="Build autonomous agents using LangGraph and FastAPI.",
        )

        self.assertEqual(result, note_content)
        self.gateway._call_gemini.assert_called_once()
        cache_key = "personal_note:cybertech:ai engineer"
        cached = getattr(self.cache, "get_answer", self.cache.get_cached_answer)(cache_key)
        self.assertEqual(cached, note_content)

    async def test_generate_personal_note_ollama_fallback(self) -> None:
        """Verify fallback to Ollama when Gemini fails or is unconfigured."""
        self.gateway.gemini_api_key = None
        self.gateway.openrouter_api_key = None
        ollama_note = "Ollama-generated tailored personal note for DataCorp."
        self.gateway._call_ollama = AsyncMock(return_value=ollama_note)

        result = await self.gateway.generate_personal_note(
            job_title="ML Engineer",
            company="DataCorp",
        )

        self.assertEqual(result, ollama_note)
        self.gateway._call_ollama.assert_called_once()

    async def test_generate_personal_note_offline_fallback(self) -> None:
        """Verify offline template fallback with candidate profile and company/title."""
        self.gateway.gemini_api_key = None
        self.gateway.openrouter_api_key = None
        self.gateway._call_ollama = AsyncMock(side_effect=httpx.ConnectError("Ollama offline"))

        profile = CandidateProfile(
            full_name="Jane Doe",
            email="jane@example.com",
            phone="+972-50-1234567",
            github_url="https://github.com/janedoe",
        )

        result = await self.gateway.generate_personal_note(
            job_title="Backend Engineer",
            company="Acme Corp",
            candidate_profile=profile,
        )

        self.assertIn("Dear Hiring Team at Acme Corp,", result)
        self.assertIn("Backend Engineer", result)
        self.assertIn("Jane Doe | jane@example.com | +972-50-1234567 | https://github.com/janedoe", result)

        # Also test with default profile (None) and unset env vars
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("CANDIDATE_")}
        with patch.dict(os.environ, clean_env, clear=True):
            default_result = await self.gateway.generate_personal_note(
                job_title="Full Stack Lead",
                company="BetaTech",
                candidate_profile=None,
            )
        self.assertIn("Dear Hiring Team at BetaTech,", default_result)
        self.assertIn("Full Stack Lead", default_result)
        banned_terms = (
            bytes.fromhex("4c696f72").decode(),
            bytes.fromhex("5a7669656c69").decode(),
            "HIT",
            bytes.fromhex("4d414720436f727073").decode(),
        )
        for banned in banned_terms:
            self.assertNotIn(banned, default_result)

    async def test_generate_personal_note_env_overrides(self) -> None:
        """Verify environment variables override default candidate details and background."""
        self.gateway.gemini_api_key = None
        self.gateway.openrouter_api_key = None
        self.gateway._call_ollama = AsyncMock(side_effect=httpx.ConnectError("Ollama offline"))

        env = {
            "CANDIDATE_NAME": "Alex Smith",
            "CANDIDATE_EMAIL": "alex@example.org",
            "CANDIDATE_PHONE": "+1-555-0199",
            "CANDIDATE_GITHUB": "https://github.com/alexsmith",
            "CANDIDATE_BACKGROUND": "Experienced distributed systems engineer specializing in high-throughput data pipelines.",
        }

        with patch.dict(os.environ, env):
            result = await self.gateway.generate_personal_note(
                job_title="Systems Engineer",
                company="Omega Systems",
                candidate_profile=None,
            )

        self.assertIn("Dear Hiring Team at Omega Systems,", result)
        self.assertIn("Systems Engineer", result)
        self.assertIn("Alex Smith | alex@example.org | +1-555-0199 | https://github.com/alexsmith", result)
        self.assertIn("Experienced distributed systems engineer specializing in high-throughput data pipelines.", result)
        banned_terms = (
            bytes.fromhex("4c696f72").decode(),
            bytes.fromhex("5a7669656c69").decode(),
            "HIT",
            bytes.fromhex("4d414720436f727073").decode(),
        )
        for banned in banned_terms:
            self.assertNotIn(banned, result)

    async def test_generate_personal_note_cache_hit(self) -> None:
        """Verify second call retrieves cached personal note without network invocation."""
        first_note = "Generated note content for caching test."
        self.gateway.gemini_api_key = "fake_gemini_key"
        self.gateway._call_gemini = AsyncMock(return_value=first_note)

        # First call: cache miss, calls Gemini
        res1 = await self.gateway.generate_personal_note(
            job_title="AI Engineer",
            company="CacheCompany",
        )
        self.assertEqual(res1, first_note)
        self.assertEqual(self.gateway._call_gemini.call_count, 1)

        # Reset mock call count
        self.gateway._call_gemini.reset_mock()

        # Second call: cache hit, bypasses provider
        res2 = await self.gateway.generate_personal_note(
            job_title="AI Engineer",
            company="CacheCompany",
        )
        self.assertEqual(res2, first_note)
        self.gateway._call_gemini.assert_not_called()

    async def test_generate_personal_note_empty_company_fallback(self) -> None:
        """Verify offline template gracefully handles missing or empty company and title."""
        self.gateway.gemini_api_key = None
        self.gateway.openrouter_api_key = None
        self.gateway._call_ollama = AsyncMock(side_effect=httpx.ConnectError("Ollama offline"))

        result = await self.gateway.generate_personal_note(
            job_title="",
            company="",
            candidate_profile=None,
        )
        self.assertIn("Dear Hiring Team,\n\n", result)
        self.assertIn("enthusiasm for your team's work", result)
        self.assertIn("AI Engineer role", result)

    async def test_provider_fallback_chain_with_tokenharbor(self) -> None:
        """Verify TokenHarbor is called as primary provider ahead of Gemini."""
        question = "Do you have experience with Rust?"
        tokenharbor_answer = "Yes, built async high-throughput backend services using Tokio and Axum."

        self.gateway.tokenharbor_api_key = "fake_tokenharbor_key"
        self.gateway.gemini_api_key = "fake_gemini_key"

        self.gateway._call_tokenharbor = AsyncMock(return_value=tokenharbor_answer)
        self.gateway._call_gemini = AsyncMock()

        result = await self.gateway.ask_question(question)

        self.assertEqual(result, tokenharbor_answer)
        self.gateway._call_tokenharbor.assert_called_once()
        self.gateway._call_gemini.assert_not_called()

    def test_tokenharbor_env_configuration(self) -> None:
        """Verify environment variables correctly configure TokenHarbor."""
        with patch.dict(
            os.environ,
            {
                "TOKENHARBOR_API_KEY": "th_test_key",
                "TOKENHARBOR_BASE_URL": "https://api.tokenharbor.ai/v1",
                "TOKENHARBOR_MODEL": "deepseek-v4.1-flash:free",
            },
        ):
            gw = ResilientLLMGateway(cache=LLMCache(db_path=":memory:"))
            self.assertEqual(gw.tokenharbor_api_key, "th_test_key")
            self.assertEqual(gw.tokenharbor_base_url, "https://api.tokenharbor.ai/v1")
            self.assertEqual(gw.tokenharbor_model, "deepseek-v4.1-flash:free")


if __name__ == "__main__":
    unittest.main()

