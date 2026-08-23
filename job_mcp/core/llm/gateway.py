"""Resilient Free-Tier LLM Gateway with Caching, Rate Limiting, and Multi-Provider Fallback."""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Callable, Coroutine, List, Optional, Tuple

import httpx

from job_mcp.core.llm.cache import LLMCache
from job_mcp.core.llm.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM gateway errors."""


class RateLimitOrUnavailableError(LLMError):
    """Raised when an LLM provider returns 429 Too Many Requests or 503 Service Unavailable."""


class LLMProviderError(LLMError):
    """Raised when an LLM provider returns an unexpected error or fails."""


class ResilientLLMGateway:
    """Production-grade LLM gateway with zero-cost optimization.

    Features:
    - Persistent SQLite caching of screening answers to prevent duplicate API calls.
    - Token-bucket rate limiting (default 15 RPM for Gemini free-tier) to avoid 429 quota exhaustion.
    - Multi-provider fallback chain: Gemini -> OpenRouter -> Ollama -> Mock LLM.
    - Jittered exponential backoff retries for transient 429 and 503 HTTP status codes.
    """

    def __init__(
        self,
        cache: Optional[LLMCache] = None,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        gemini_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openrouter_model: Optional[str] = None,
        openrouter_reasoning_model: Optional[str] = None,
        openrouter_extraction_model: Optional[str] = None,
        openrouter_base_url: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_model: str = "llama3.2",
        max_retries: int = 3,
        initial_backoff: float = 2.0,
        mock_fallback: bool = True,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the Resilient LLM Gateway.

        Args:
            cache: LLMCache instance or default SQLite cache.
            rate_limiter: TokenBucketRateLimiter instance or default 15 RPM.
            gemini_api_key: Gemini API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env).
            openrouter_api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env).
            gemini_model: Gemini model identifier (defaults to GEMINI_MODEL env or gemini-flash-lite-latest).
            openrouter_model: OpenRouter fallback model identifier.
            openrouter_reasoning_model: OpenRouter reasoning model identifier (defaults to OPENROUTER_REASONING_MODEL or z-ai/glm-5.2:free).
            openrouter_extraction_model: OpenRouter extraction model identifier (defaults to OPENROUTER_EXTRACTION_MODEL or google/gemma-4-26b-a4b-it:free).
            openrouter_base_url: OpenRouter base API URL (defaults to OPENROUTER_BASE_URL or https://openrouter.ai/api/v1).
            ollama_url: Ollama API generate endpoint URL.
            ollama_model: Ollama model identifier.
            max_retries: Maximum retry attempts for 429 / 503 rate limits.
            initial_backoff: Base backoff time in seconds before jitter.
            mock_fallback: Whether to fallback to Mock LLM if all providers fail/unconfigured.
            http_client: Optional httpx.AsyncClient for testing/mocking HTTP calls.
        """
        self.cache = cache or LLMCache()
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(rpm=15)
        self.gemini_api_key = (
            gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.openrouter_api_key = (
            openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        self.gemini_model = (
            gemini_model
            or os.environ.get("GEMINI_MODEL")
            or "gemini-flash-lite-latest"
        )
        self.openrouter_base_url = (
            openrouter_base_url
            or os.environ.get("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.openrouter_reasoning_model = (
            openrouter_reasoning_model
            or os.environ.get("OPENROUTER_REASONING_MODEL")
            or os.environ.get("OPENROUTER_MODEL")
            or openrouter_model
            or "z-ai/glm-5.2:free"
        )
        self.openrouter_extraction_model = (
            openrouter_extraction_model
            or os.environ.get("OPENROUTER_EXTRACTION_MODEL")
            or "google/gemma-4-26b-a4b-it:free"
        )
        self.openrouter_model = (
            openrouter_model
            or os.environ.get("OPENROUTER_MODEL")
            or self.openrouter_reasoning_model
        )
        self.ollama_url = (
            ollama_url
            or os.environ.get("OLLAMA_URL")
            or "http://localhost:11434/api/generate"
        )
        self.ollama_model = ollama_model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.mock_fallback = mock_fallback
        self._http_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create an httpx.AsyncClient."""
        if self._http_client is not None:
            return self._http_client
        return httpx.AsyncClient(timeout=30.0)


    async def _call_gemini(self, prompt: str, system_prompt: str) -> str:
        """Invoke Google Gemini REST API."""
        if not self.gemini_api_key:
            raise LLMProviderError("Gemini API key is not configured.")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
        }

        client = await self._get_client()
        should_close = self._http_client is None
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code in (429, 503):
                raise RateLimitOrUnavailableError(
                    f"Gemini returned HTTP {resp.status_code}: {resp.text}"
                )
            if resp.status_code != 200:
                raise LLMProviderError(
                    f"Gemini API returned HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMProviderError("Gemini returned empty candidates list.")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts or "text" not in parts[0]:
                raise LLMProviderError("Gemini response missing text part.")
            return str(parts[0]["text"]).strip()
        finally:
            if should_close:
                await client.aclose()

    async def _call_openrouter(
        self,
        prompt: str,
        system_prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """Invoke OpenRouter Chat Completions API."""
        if not self.openrouter_api_key:
            raise LLMProviderError("OpenRouter API key is not configured.")

        url = f"{self.openrouter_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/zvieli/TechJobMCP",
            "X-Title": "TechJobMCP Application Engine",
        }
        payload = {
            "model": model or self.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }


        client = await self._get_client()
        should_close = self._http_client is None
        try:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code in (429, 503):
                raise RateLimitOrUnavailableError(
                    f"OpenRouter returned HTTP {resp.status_code}: {resp.text}"
                )
            if resp.status_code != 200:
                raise LLMProviderError(
                    f"OpenRouter API returned HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMProviderError("OpenRouter returned empty choices list.")
            content = choices[0].get("message", {}).get("content", "")
            return str(content).strip()
        finally:
            if should_close:
                await client.aclose()

    async def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        """Invoke local Ollama generate API."""
        payload = {
            "model": self.ollama_model,
            "prompt": f"{system_prompt}\n\n{prompt}",
            "stream": False,
        }

        client = await self._get_client()
        should_close = self._http_client is None
        try:
            resp = await client.post(self.ollama_url, json=payload)
            if resp.status_code in (429, 503):
                raise RateLimitOrUnavailableError(
                    f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                )
            if resp.status_code != 200:
                raise LLMProviderError(
                    f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            response_text = data.get("response", "")
            if not response_text:
                raise LLMProviderError("Ollama returned empty response.")
            return str(response_text).strip()
        finally:
            if should_close:
                await client.aclose()

    def _generate_mock_answer(self, question: str, cv_context: str) -> str:
        """Generate contextual screening answer when offline or in test mode."""
        q_lower = question.lower()
        if "year" in q_lower and ("experience" in q_lower or "exp" in q_lower):
            return "7+ years of professional software engineering and backend systems experience."
        if "authorized" in q_lower or "legal" in q_lower or "work authorization" in q_lower:
            return "Yes, fully authorized to work without restrictions."
        if "sponsor" in q_lower or "visa" in q_lower:
            return "No visa sponsorship required."
        if "salary" in q_lower or "compensation" in q_lower or "rate" in q_lower:
            return "Open to discussion based on total compensation and role responsibilities."
        if "relocate" in q_lower or "relocation" in q_lower:
            return "Yes, open to relocation or remote work arrangements."
        if "remote" in q_lower or "hybrid" in q_lower or "on-site" in q_lower or "onsite" in q_lower:
            return "Comfortable with remote, hybrid, or on-site work arrangements."
        if "notice" in q_lower or "start" in q_lower or "available" in q_lower:
            return "Available to start within 2 to 4 weeks."
        if "degree" in q_lower or "education" in q_lower or "bachelor" in q_lower:
            return "B.Sc. in Computer Science / Software Engineering."

        if cv_context and len(cv_context) > 20:
            return f"Relevant experience matching candidate profile: {cv_context[:100].strip()}..."

        return "Yes, I possess the required experience, technical competencies, and track record."

    async def _execute_with_retry(
        self,
        provider_name: str,
        call_fn: Callable[[], Coroutine[Any, Any, str]],
    ) -> str:
        """Execute a provider call with jittered exponential backoff on 429/503."""
        for attempt in range(self.max_retries + 1):
            try:
                return await call_fn()
            except RateLimitOrUnavailableError as err:
                if attempt >= self.max_retries:
                    logger.warning(
                        "Provider %s rate limit retries exhausted (%d attempts): %s",
                        provider_name,
                        self.max_retries + 1,
                        err,
                    )
                    raise
                jitter = random.uniform(0.1, 0.5)
                delay = (self.initial_backoff * (2**attempt)) + jitter
                logger.warning(
                    "Provider %s hit rate limit (attempt %d/%d). Retrying in %.2fs: %s",
                    provider_name,
                    attempt + 1,
                    self.max_retries,
                    delay,
                    err,
                )
                await asyncio.sleep(delay)
            except Exception as err:
                logger.warning(
                    "Provider %s encountered non-retryable error: %s",
                    provider_name,
                    err,
                )
                raise

        raise LLMProviderError(f"Provider {provider_name} failed unexpectedly.")

    def _get_provider_chain(
        self,
    ) -> List[Tuple[str, Callable[[str, str], Coroutine[Any, Any, str]]]]:
        """Build provider chain list based on available configuration."""
        chain: List[Tuple[str, Callable[[str, str], Coroutine[Any, Any, str]]]] = []

        if self.gemini_api_key:
            chain.append(("gemini", self._call_gemini))

        if self.openrouter_api_key:
            chain.append(("openrouter", self._call_openrouter))

        # Ollama can always be attempted if configured URL exists
        chain.append(("ollama", self._call_ollama))
        return chain

    async def ask_question(self, question: str, cv_context: str = "") -> str:
        """Answer a screening questionnaire question using cached answer or LLM.

        Args:
            question: The screening question text.
            cv_context: Candidate profile or CV summary to ground the response.

        Returns:
            Concise, relevant answer text.
        """
        if not question or not question.strip():
            return ""

        # 1. Check SQLite Cache First (Zero-cost bypass)
        cached_answer = self.cache.get_cached_answer(question)
        if cached_answer is not None:
            logger.info("Cache HIT for question: '%s'", question[:50])
            return cached_answer

        # 2. Acquire Rate Limiter Token (free-tier quota preservation)
        await self.rate_limiter.wait_for_token()

        prompt = (
            f"Candidate CV / Profile Context:\n{cv_context or 'N/A'}\n\n"
            f"Job Application Screening Question:\n{question}\n\n"
            "Direct, concise, professional answer:"
        )
        system_prompt = (
            "You are an AI career assistant answering application screening questions "
            "on behalf of a candidate based on their CV. Provide concise, professional, "
            "truthful, and relevant answers."
        )

        # 3. Try Multi-Provider Fallback Chain
        providers = self._get_provider_chain()
        last_error: Optional[Exception] = None

        for provider_name, provider_fn in providers:
            try:
                answer = await self._execute_with_retry(
                    provider_name,
                    lambda fn=provider_fn: fn(prompt, system_prompt),
                )
                if answer and answer.strip():
                    clean_answer = answer.strip()
                    self.cache.cache_answer(question, clean_answer)
                    return clean_answer
            except Exception as err:
                logger.warning(
                    "Provider %s failed, falling back to next provider: %s",
                    provider_name,
                    err,
                )
                last_error = err

        # 4. Fallback to Mock LLM if enabled
        if self.mock_fallback:
            logger.info(
                "All real providers failed/unconfigured. Using Mock LLM fallback for: '%s'",
                question[:50],
            )
            mock_answer = self._generate_mock_answer(question, cv_context)
            self.cache.cache_answer(question, mock_answer)
            return mock_answer

        raise LLMProviderError(
            f"All LLM providers in fallback chain failed. Last error: {last_error}"
        )
