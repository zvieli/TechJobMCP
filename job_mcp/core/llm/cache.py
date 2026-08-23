"""SQLite-backed caching for LLM screening questionnaire answers."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class LLMCache:
    """Persistent SQLite cache for screening questionnaire answers.

    Normalizes and hashes questions to avoid redundant LLM invocations and
    drastically reduce latency and API consumption.
    """

    def __init__(self, db_path: str = "llm_cache.db") -> None:
        """Initialize the SQLite cache.

        Args:
            db_path: Filepath for the SQLite database or ':memory:'.
        """
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._mem_conn: Optional[sqlite3.Connection] = None

        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._mem_conn is not None:
            return self._mem_conn

        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Create answer_cache table if not exists."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS answer_cache (
                        question_hash TEXT PRIMARY KEY,
                        question_text TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                conn.commit()
            finally:
                if self._mem_conn is None:
                    conn.close()

    @staticmethod
    def normalize_question(question: str) -> str:
        """Normalize question text by lowercasing and stripping whitespace."""
        if not question:
            return ""
        return " ".join(question.strip().lower().split())

    @classmethod
    def hash_question(cls, question: str) -> str:
        """Generate SHA-256 hash of normalized question text."""
        normalized = cls.normalize_question(question)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get_cached_answer(self, question: str) -> Optional[str]:
        """Fetch cached answer for a question if available.

        Args:
            question: Screening question text.

        Returns:
            Cached answer string if found, None otherwise.
        """
        if not question or not question.strip():
            return None

        q_hash = self.hash_question(question)
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT answer_text FROM answer_cache WHERE question_hash = ?",
                    (q_hash,),
                )
                row = cursor.fetchone()
                if row:
                    logger.debug("Cache hit for question hash: %s", q_hash[:8])
                    return str(row["answer_text"])
            finally:
                if self._mem_conn is None:
                    conn.close()

        logger.debug("Cache miss for question hash: %s", q_hash[:8])
        return None

    def cache_answer(self, question: str, answer: str) -> None:
        """Store or update question-answer pair in SQLite cache.

        Args:
            question: Screening question text.
            answer: Answer text to cache.
        """
        if not question or not question.strip() or answer is None:
            return

        q_hash = self.hash_question(question)
        normalized_q = self.normalize_question(question)
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO answer_cache (question_hash, question_text, answer_text, timestamp)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(question_hash) DO UPDATE SET
                        answer_text = excluded.answer_text,
                        question_text = excluded.question_text,
                        timestamp = CURRENT_TIMESTAMP;
                    """,
                    (q_hash, normalized_q, answer.strip()),
                )
                conn.commit()
                logger.debug("Cached answer for question hash: %s", q_hash[:8])
            finally:
                if self._mem_conn is None:
                    conn.close()

    def clear(self) -> None:
        """Clear all cached answers."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM answer_cache;")
                conn.commit()
            finally:
                if self._mem_conn is None:
                    conn.close()

    def size(self) -> int:
        """Return the total number of cached answers."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) AS total FROM answer_cache;")
                row = cursor.fetchone()
                return int(row["total"]) if row else 0
            finally:
                if self._mem_conn is None:
                    conn.close()
