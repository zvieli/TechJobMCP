"""SQLite-backed application ledger and duplicate submission guardrails."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional

from job_mcp.models.ledger import ApplicationEntry, ApplicationMethod, ApplicationStatus

logger = logging.getLogger(__name__)


class ApplicationLedger:
    """Persistent audit ledger for job applications with duplicate prevention."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the application ledger.

        Args:
            db_path: Path to SQLite DB file or ':memory:'. If None, resolves from
                     APPLICATION_LEDGER_PATH environment variable or defaults to
                     'application_ledger.db'.
        """
        if db_path is None:
            db_path = os.getenv("APPLICATION_LEDGER_PATH", "application_ledger.db")

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
        """Get an active SQLite connection."""
        if self._mem_conn is not None:
            return self._mem_conn

        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Create applications table and indexes if not existing."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS applications (
                        job_id TEXT NOT NULL,
                        company TEXT NOT NULL,
                        job_title TEXT NOT NULL,
                        source TEXT DEFAULT '',
                        applied_at TEXT NOT NULL,
                        method TEXT NOT NULL,
                        status TEXT NOT NULL,
                        match_score REAL,
                        cv_used TEXT,
                        response_payload TEXT,
                        error_message TEXT,
                        notes TEXT
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_applications_job_id ON applications(job_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_applications_applied_at ON applications(applied_at);"
                )
                conn.commit()
            finally:
                if self._mem_conn is None:
                    conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> ApplicationEntry:
        """Convert a database row into an ApplicationEntry model."""
        payload = row["response_payload"]
        if payload is not None:
            try:
                payload = json.loads(payload)
            except Exception:
                pass

        try:
            applied_at = datetime.fromisoformat(row["applied_at"])
        except Exception:
            applied_at = row["applied_at"]

        return ApplicationEntry(
            job_id=str(row["job_id"]),
            company=str(row["company"]),
            job_title=str(row["job_title"]),
            source=str(row["source"] or ""),
            applied_at=applied_at,
            method=str(row["method"]),
            status=str(row["status"]),
            match_score=float(row["match_score"]) if row["match_score"] is not None else None,
            cv_used=str(row["cv_used"]) if row["cv_used"] is not None else None,
            response_payload=payload,
            error_message=str(row["error_message"]) if row["error_message"] is not None else None,
            notes=str(row["notes"]) if row["notes"] is not None else None,
        )

    def record_application(self, entry: ApplicationEntry) -> None:
        """Record an application entry in the ledger.

        Args:
            entry: The application entry to persist.
        """
        applied_at_str = (
            entry.applied_at.isoformat()
            if isinstance(entry.applied_at, datetime)
            else str(entry.applied_at)
        )
        method_str = entry.method.value if isinstance(entry.method, Enum) else str(entry.method)
        status_str = entry.status.value if isinstance(entry.status, Enum) else str(entry.status)

        payload_str: Optional[str] = None
        if isinstance(entry.response_payload, (dict, list)):
            payload_str = json.dumps(entry.response_payload)
        elif entry.response_payload is not None:
            payload_str = str(entry.response_payload)

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO applications (
                        job_id, company, job_title, source, applied_at,
                        method, status, match_score, cv_used,
                        response_payload, error_message, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        entry.job_id,
                        entry.company,
                        entry.job_title,
                        entry.source,
                        applied_at_str,
                        method_str,
                        status_str,
                        entry.match_score,
                        entry.cv_used,
                        payload_str,
                        entry.error_message,
                        entry.notes,
                    ),
                )
                conn.commit()
                logger.debug("Recorded application for job_id %s with status %s", entry.job_id, status_str)
            finally:
                if self._mem_conn is None:
                    conn.close()

    def is_applied(self, job_id: str) -> bool:
        """Check if a job has already been successfully applied to.

        Args:
            job_id: The job identifier.

        Returns:
            True if a successful application exists for this job, False otherwise.
        """
        if not job_id:
            return False

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) AS total FROM applications
                    WHERE job_id = ? AND status = ?;
                    """,
                    (job_id, ApplicationStatus.SUCCESS.value),
                )
                row = cursor.fetchone()
                return bool(row and row["total"] > 0)
            finally:
                if self._mem_conn is None:
                    conn.close()

    def get_application(self, job_id: str) -> Optional[ApplicationEntry]:
        """Fetch the most recent application entry for a job.

        Args:
            job_id: The job identifier.

        Returns:
            ApplicationEntry if found, None otherwise.
        """
        if not job_id:
            return None

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM applications
                    WHERE job_id = ?
                    ORDER BY applied_at DESC
                    LIMIT 1;
                    """,
                    (job_id,),
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_entry(row)
                return None
            finally:
                if self._mem_conn is None:
                    conn.close()

    def get_daily_count(self, date_str: Optional[str] = None) -> int:
        """Count successful applications on a specific date (UTC).

        Args:
            date_str: Target date in 'YYYY-MM-DD' format. If None, defaults to current UTC date.

        Returns:
            Number of successful applications on that date.
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            date_str = str(date_str).strip()[:10]

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) AS total FROM applications
                    WHERE status = ? AND substr(applied_at, 1, 10) = ?;
                    """,
                    (ApplicationStatus.SUCCESS.value, date_str),
                )
                row = cursor.fetchone()
                return int(row["total"]) if row else 0
            finally:
                if self._mem_conn is None:
                    conn.close()

    def list_applications(
        self,
        limit: int = 50,
        status: Optional[str | ApplicationStatus] = None,
    ) -> list[ApplicationEntry]:
        """List recorded applications ordered by most recent first.

        Args:
            limit: Maximum number of entries to return.
            status: Optional status filter.

        Returns:
            List of ApplicationEntry models.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if status is not None:
                    status_val = status.value if isinstance(status, Enum) else str(status)
                    cursor.execute(
                        """
                        SELECT * FROM applications
                        WHERE status = ?
                        ORDER BY applied_at DESC
                        LIMIT ?;
                        """,
                        (status_val, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM applications
                        ORDER BY applied_at DESC
                        LIMIT ?;
                        """,
                        (limit,),
                    )
                rows = cursor.fetchall()
                return [self._row_to_entry(row) for row in rows]
            finally:
                if self._mem_conn is None:
                    conn.close()

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        with self._lock:
            if self._mem_conn is not None:
                try:
                    self._mem_conn.close()
                except Exception:
                    pass
                self._mem_conn = None
