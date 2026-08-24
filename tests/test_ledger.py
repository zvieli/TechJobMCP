"""Unit tests for ApplicationLedger and safety guardrail models."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import pytest
import sqlite3

from job_mcp.core.application.ledger_service import ApplicationLedger
from job_mcp.models.ledger import (
    ApplicationEntry,
    ApplicationMethod,
    ApplicationStatus,
)


@pytest.fixture
def memory_ledger():
    """Create an in-memory ApplicationLedger instance."""
    ledger = ApplicationLedger(db_path=":memory:")
    yield ledger
    ledger.close()


@pytest.fixture
def tmp_disk_ledger(tmp_path: Path):
    """Create a temporary disk-backed ApplicationLedger instance."""
    db_file = tmp_path / "subdir" / "test_ledger.db"
    ledger = ApplicationLedger(db_path=str(db_file))
    yield ledger, db_file
    ledger.close()


def test_models_and_enums():
    """Test ApplicationStatus, ApplicationMethod, and ApplicationEntry creation."""
    assert ApplicationStatus.SUCCESS == "success"
    assert ApplicationStatus.FAILED == "failed"
    assert ApplicationStatus.STAGED == "staged"
    assert ApplicationStatus.BLOCKED == "blocked"

    assert ApplicationMethod.API == "api"
    assert ApplicationMethod.EASY_APPLY == "easy_apply"
    assert ApplicationMethod.BROWSER == "browser"

    entry = ApplicationEntry(
        job_id="job-123",
        company="TechCorp",
        job_title="Senior Python Engineer",
        method=ApplicationMethod.API,
        status=ApplicationStatus.SUCCESS,
        match_score=0.95,
        cv_used="/path/to/cv.pdf",
        response_payload={"submission_id": "sub-999"},
        notes="Applied automatically",
    )
    assert entry.job_id == "job-123"
    assert entry.company == "TechCorp"
    assert entry.job_title == "Senior Python Engineer"
    assert entry.source == ""
    assert isinstance(entry.applied_at, datetime)
    assert entry.method == ApplicationMethod.API
    assert entry.status == ApplicationStatus.SUCCESS
    assert entry.match_score == 0.95
    assert entry.response_payload == {"submission_id": "sub-999"}
    assert entry.notes == "Applied automatically"


def test_schema_creation_in_memory(memory_ledger: ApplicationLedger):
    """Test schema and indexes creation in :memory: database."""
    conn = memory_ledger._get_connection()
    cursor = conn.cursor()

    # Verify table existence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applications';")
    assert cursor.fetchone() is not None

    # Verify indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='applications';")
    indexes = {row["name"] for row in cursor.fetchall()}
    assert "idx_applications_job_id" in indexes
    assert "idx_applications_status" in indexes
    assert "idx_applications_applied_at" in indexes


def test_schema_creation_on_disk(tmp_disk_ledger):
    """Test schema and directory creation on disk."""
    ledger, db_file = tmp_disk_ledger
    assert db_file.exists()

    entry = ApplicationEntry(
        job_id="job-disk-1",
        company="DiskCorp",
        job_title="Database Engineer",
        method="browser",
        status="success",
    )
    ledger.record_application(entry)

    # Open direct connection to disk file to verify persistence
    direct_conn = sqlite3.connect(str(db_file))
    cursor = direct_conn.cursor()
    cursor.execute("SELECT job_id, company FROM applications WHERE job_id = 'job-disk-1';")
    row = cursor.fetchone()
    assert row == ("job-disk-1", "DiskCorp")
    direct_conn.close()


def test_env_var_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test resolving database path from APPLICATION_LEDGER_PATH env var."""
    env_db = tmp_path / "env_ledger.db"
    monkeypatch.setenv("APPLICATION_LEDGER_PATH", str(env_db))

    ledger = ApplicationLedger()
    assert ledger.db_path == str(env_db)
    assert env_db.exists()
    ledger.close()


def test_record_and_get_application(memory_ledger: ApplicationLedger):
    """Test recording and retrieving an application entry."""
    assert memory_ledger.get_application("job-unknown") is None

    entry = ApplicationEntry(
        job_id="job-456",
        company="Alpha AI",
        job_title="ML Engineer",
        source="comeet",
        applied_at=datetime(2026, 8, 23, 12, 0, 0, tzinfo=timezone.utc),
        method=ApplicationMethod.EASY_APPLY,
        status=ApplicationStatus.SUCCESS,
        match_score=0.88,
        cv_used="resume_ml.pdf",
        response_payload={"status": "submitted", "id": 101},
        error_message=None,
        notes="Automated run",
    )
    memory_ledger.record_application(entry)

    fetched = memory_ledger.get_application("job-456")
    assert fetched is not None
    assert fetched.job_id == "job-456"
    assert fetched.company == "Alpha AI"
    assert fetched.job_title == "ML Engineer"
    assert fetched.source == "comeet"
    assert fetched.method == "easy_apply"
    assert fetched.status == "success"
    assert fetched.match_score == 0.88
    assert fetched.cv_used == "resume_ml.pdf"
    assert fetched.response_payload == {"status": "submitted", "id": 101}
    assert fetched.error_message is None
    assert fetched.notes == "Automated run"


def test_get_application_returns_latest_first(memory_ledger: ApplicationLedger):
    """Test get_application returns the latest entry if multiple attempts exist."""
    t1 = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 21, 10, 0, 0, tzinfo=timezone.utc)

    e1 = ApplicationEntry(
        job_id="job-multi",
        company="MultiCorp",
        job_title="DevOps",
        applied_at=t1,
        method="api",
        status="failed",
        error_message="Network timeout",
    )
    e2 = ApplicationEntry(
        job_id="job-multi",
        company="MultiCorp",
        job_title="DevOps",
        applied_at=t2,
        method="api",
        status="success",
    )
    memory_ledger.record_application(e1)
    memory_ledger.record_application(e2)

    latest = memory_ledger.get_application("job-multi")
    assert latest is not None
    assert latest.status == "success"
    assert latest.applied_at.isoformat().startswith("2026-08-21")


def test_is_applied_guardrail(memory_ledger: ApplicationLedger):
    """Test is_applied accurately reports successful vs failed/staged applications."""
    assert memory_ledger.is_applied("job-999") is False
    assert memory_ledger.is_applied("") is False

    # Failed application should NOT count as applied
    failed_entry = ApplicationEntry(
        job_id="job-failed",
        company="FailInc",
        job_title="Tester",
        method="browser",
        status=ApplicationStatus.FAILED,
        error_message="Selector missing",
    )
    memory_ledger.record_application(failed_entry)
    assert memory_ledger.is_applied("job-failed") is False

    # Staged application should NOT count as applied
    staged_entry = ApplicationEntry(
        job_id="job-staged",
        company="StageInc",
        job_title="Designer",
        method="api",
        status=ApplicationStatus.STAGED,
    )
    memory_ledger.record_application(staged_entry)
    assert memory_ledger.is_applied("job-staged") is False

    # Blocked application should NOT count as applied
    blocked_entry = ApplicationEntry(
        job_id="job-blocked",
        company="BlockInc",
        job_title="Security Engineer",
        method="api",
        status=ApplicationStatus.BLOCKED,
    )
    memory_ledger.record_application(blocked_entry)
    assert memory_ledger.is_applied("job-blocked") is False

    # Success application DOES count as applied
    success_entry = ApplicationEntry(
        job_id="job-success",
        company="SuccessInc",
        job_title="Architect",
        method="easy_apply",
        status=ApplicationStatus.SUCCESS,
    )
    memory_ledger.record_application(success_entry)
    assert memory_ledger.is_applied("job-success") is True
    # Test company and job title matching
    assert memory_ledger.is_applied("different-id", company="SuccessInc", job_title="Architect") is True
    assert memory_ledger.is_applied("", company="SuccessInc", job_title="Architect") is True
    assert memory_ledger.is_applied("", company="SuccessInc", job_title="OtherRole") is False
    assert memory_ledger.is_applied("", company="OtherCo", job_title="Architect") is False


def test_get_daily_count_across_dates_and_statuses(memory_ledger: ApplicationLedger):
    """Test get_daily_count filters by UTC date and status=success."""
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)
    today_str = today.strftime("%Y-%m-%d")
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    # Add 2 successes today, 1 failure today
    memory_ledger.record_application(
        ApplicationEntry(
            job_id="j-today-1",
            company="Co1",
            job_title="Role1",
            applied_at=today,
            method="api",
            status=ApplicationStatus.SUCCESS,
        )
    )
    memory_ledger.record_application(
        ApplicationEntry(
            job_id="j-today-2",
            company="Co2",
            job_title="Role2",
            applied_at=today,
            method="api",
            status=ApplicationStatus.SUCCESS,
        )
    )
    memory_ledger.record_application(
        ApplicationEntry(
            job_id="j-today-3",
            company="Co3",
            job_title="Role3",
            applied_at=today,
            method="api",
            status=ApplicationStatus.FAILED,
        )
    )

    # Add 1 success yesterday
    memory_ledger.record_application(
        ApplicationEntry(
            job_id="j-yesterday-1",
            company="Co4",
            job_title="Role4",
            applied_at=yesterday,
            method="api",
            status=ApplicationStatus.SUCCESS,
        )
    )

    # Count today (default and explicit)
    assert memory_ledger.get_daily_count() == 2
    assert memory_ledger.get_daily_count(today_str) == 2

    # Count yesterday
    assert memory_ledger.get_daily_count(yesterday_str) == 1

    # Count future date with no applications
    assert memory_ledger.get_daily_count("2099-01-01") == 0


def test_list_applications_filtering_and_limits(memory_ledger: ApplicationLedger):
    """Test list_applications filtering by status and respecting limits."""
    for i in range(10):
        status = ApplicationStatus.SUCCESS if i % 2 == 0 else ApplicationStatus.FAILED
        memory_ledger.record_application(
            ApplicationEntry(
                job_id=f"job-list-{i}",
                company=f"Company {i}",
                job_title=f"Title {i}",
                method="api",
                status=status,
                applied_at=datetime(2026, 8, 1, 10, i, 0, tzinfo=timezone.utc),
            )
        )

    # Total 10 entries (5 success, 5 failed)
    all_entries = memory_ledger.list_applications(limit=50)
    assert len(all_entries) == 10
    # Ordered DESC by applied_at
    assert all_entries[0].job_id == "job-list-9"
    assert all_entries[-1].job_id == "job-list-0"

    # Limit = 3
    top3 = memory_ledger.list_applications(limit=3)
    assert len(top3) == 3
    assert [e.job_id for e in top3] == ["job-list-9", "job-list-8", "job-list-7"]

    # Filter by success
    successes = memory_ledger.list_applications(limit=50, status=ApplicationStatus.SUCCESS)
    assert len(successes) == 5
    assert all(e.status == "success" for e in successes)

    # Filter by failed as string
    failures = memory_ledger.list_applications(limit=50, status="failed")
    assert len(failures) == 5
    assert all(e.status == "failed" for e in failures)


def test_string_response_payload(memory_ledger: ApplicationLedger):
    """Test non-json string response_payload preservation."""
    entry = ApplicationEntry(
        job_id="job-str-payload",
        company="StrCo",
        job_title="Engineer",
        method="browser",
        status="success",
        response_payload="Raw string response from webhook",
    )
    memory_ledger.record_application(entry)

    fetched = memory_ledger.get_application("job-str-payload")
    assert fetched is not None
    assert fetched.response_payload == "Raw string response from webhook"


def test_concurrent_access_thread_safety(tmp_path: Path):
    """Test multi-threaded concurrent inserts and reads."""
    db_file = tmp_path / "concurrent_ledger.db"
    ledger = ApplicationLedger(db_path=str(db_file))

    def worker(worker_id: int):
        for i in range(20):
            job_id = f"worker-{worker_id}-job-{i}"
            ledger.record_application(
                ApplicationEntry(
                    job_id=job_id,
                    company=f"WorkerCorp {worker_id}",
                    job_title="Concurrent Dev",
                    method="api",
                    status=ApplicationStatus.SUCCESS,
                )
            )
            assert ledger.is_applied(job_id) is True

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in futures:
            f.result()

    assert len(ledger.list_applications(limit=200)) == 160
    assert ledger.get_daily_count() == 160
    ledger.close()
