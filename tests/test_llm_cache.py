"""Unit tests for SQLite-backed LLMCache."""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
import unittest

from job_mcp.core.llm.cache import LLMCache


class TestLLMCache(unittest.TestCase):
    """Test suite for LLMCache functionality, normalization, and concurrency."""

    def setUp(self) -> None:
        """Create a temporary database file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_cache.db")
        self.cache = LLMCache(db_path=self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_cache_miss(self) -> None:
        """Verify cache returns None when a question is not found."""
        self.assertIsNone(self.cache.get_cached_answer("Unknown screening question?"))
        self.assertIsNone(self.cache.get_cached_answer(""))
        self.assertIsNone(self.cache.get_cached_answer("   "))

    def test_cache_hit_and_normalization(self) -> None:
        """Verify cached answers are retrieved with whitespace and case normalization."""
        question_original = "How many years of Python experience do you have?"
        answer = "Over 6 years of Python experience building backend systems."

        self.cache.cache_answer(question_original, answer)
        self.assertEqual(self.cache.size(), 1)

        # Exact match
        self.assertEqual(self.cache.get_cached_answer(question_original), answer)

        # Uppercase match
        self.assertEqual(
            self.cache.get_cached_answer("HOW MANY YEARS OF PYTHON EXPERIENCE DO YOU HAVE?"),
            answer,
        )

        # Extra whitespace and newline match
        self.assertEqual(
            self.cache.get_cached_answer("  how  many   years of \n python experience do you have?  "),
            answer,
        )

    def test_cache_overwrite(self) -> None:
        """Verify caching an existing question updates the answer."""
        question = "What is your target salary?"
        self.cache.cache_answer(question, "120,000 USD")
        self.assertEqual(self.cache.get_cached_answer(question), "120,000 USD")

        self.cache.cache_answer(question, "140,000 USD")
        self.assertEqual(self.cache.get_cached_answer(question), "140,000 USD")
        self.assertEqual(self.cache.size(), 1)

    def test_cache_clear_and_size(self) -> None:
        """Verify clear() empties the cache and size() reports accurate counts."""
        self.assertEqual(self.cache.size(), 0)
        self.cache.cache_answer("Question 1", "Answer 1")
        self.cache.cache_answer("Question 2", "Answer 2")
        self.assertEqual(self.cache.size(), 2)

        self.cache.clear()
        self.assertEqual(self.cache.size(), 0)
        self.assertIsNone(self.cache.get_cached_answer("Question 1"))

    def test_in_memory_db(self) -> None:
        """Verify LLMCache works with in-memory database configuration."""
        mem_cache = LLMCache(db_path=":memory:")
        mem_cache.cache_answer("Are you authorized to work?", "Yes, authorized.")
        self.assertEqual(
            mem_cache.get_cached_answer("are you authorized to work?"),
            "Yes, authorized.",
        )

    def test_concurrent_read_write(self) -> None:
        """Verify thread-safety under concurrent reads and writes."""
        def worker(idx: int) -> bool:
            q = f"Question number {idx % 5}?"
            ans = f"Answer number {idx}"
            self.cache.cache_answer(q, ans)
            res = self.cache.get_cached_answer(q)
            return res is not None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(50)]
            results = [f.result() for f in futures]

        self.assertTrue(all(results))
        self.assertEqual(self.cache.size(), 5)


if __name__ == "__main__":
    unittest.main()
