"""Unit tests for Alert Notification Engine and Job Tracker (job_mcp/notifiers/)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from job_mcp.models.schemas import Job, WorkMode


# ============================================================================
# Tests: Package Exports & BaseNotifier
# ============================================================================


class TestBaseNotifier:
    """Tests for BaseNotifier abstract class."""

    def test_base_notifier_abstract(self) -> None:
        """Verify BaseNotifier cannot be instantiated directly."""
        from job_mcp.notifiers.base import BaseNotifier

        with pytest.raises(TypeError):
            BaseNotifier()  # type: ignore[abstract]

    def test_concrete_notifier_subclass(self) -> None:
        """Verify concrete notifier implementing all abstract methods works."""
        from job_mcp.notifiers.base import BaseNotifier

        class DummyNotifier(BaseNotifier):
            def format_alert(self, jobs: list[Job], title: str | None = None) -> str:
                return f"Alert: {len(jobs)} jobs"

            async def send_alert(self, jobs: list[Job], title: str | None = None) -> bool:
                return True

            async def check_health(self) -> bool:
                return True

        notifier = DummyNotifier()
        assert notifier.format_alert([]) == "Alert: 0 jobs"


# ============================================================================
# Tests: JobTracker
# ============================================================================


class TestJobTracker:
    """Tests for JobTracker seen state tracking and persistence."""

    @pytest.fixture
    def sample_jobs(self) -> list[Job]:
        return [
            Job(
                job_id="job_001",
                title="Python Backend Engineer",
                company="TechCorp",
                location="Tel Aviv",
                url="https://example.com/job/1",
            ),
            Job(
                job_id="job_002",
                title="Frontend Developer",
                company="WebCo",
                location="Remote",
                url="https://example.com/job/2",
            ),
            Job(
                job_id="job_003",
                title="DevOps Lead",
                company="CloudTech",
                location="Haifa",
                url="https://example.com/job/3",
            ),
        ]

    def test_in_memory_tracker_init(self) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        tracker = JobTracker()
        assert len(tracker) == 0
        stats = tracker.get_stats()
        assert stats["total_seen"] == 0
        assert stats["storage_path"] is None
        assert stats["auto_save"] is False or stats["auto_save"] is True

    def test_mark_and_is_seen_with_job(self, sample_jobs: list[Job]) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        tracker = JobTracker()
        job = sample_jobs[0]

        assert not tracker.is_seen(job)
        assert job not in tracker

        # First mark returns True (newly added)
        assert tracker.mark_seen(job) is True
        assert tracker.is_seen(job) is True
        assert job in tracker
        assert len(tracker) == 1

        # Second mark returns False (already seen)
        assert tracker.mark_seen(job) is False
        assert len(tracker) == 1

    def test_mark_and_is_seen_with_string_id(self) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        tracker = JobTracker()
        assert not tracker.is_seen("custom_id_123")
        assert tracker.mark_seen("custom_id_123") is True
        assert tracker.is_seen("custom_id_123") is True
        assert tracker.mark_seen("custom_id_123") is False

    def test_filter_unseen(self, sample_jobs: list[Job]) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        tracker = JobTracker()
        tracker.mark_seen(sample_jobs[0])

        unseen = tracker.filter_unseen(sample_jobs)
        assert len(unseen) == 2
        assert sample_jobs[1] in unseen
        assert sample_jobs[2] in unseen
        assert sample_jobs[0] not in unseen
        # Original tracker seen state didn't change with auto_mark=False
        assert len(tracker) == 1

    def test_filter_unseen_with_auto_mark(self, sample_jobs: list[Job]) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        tracker = JobTracker()
        unseen = tracker.filter_unseen(sample_jobs, auto_mark=True)
        assert len(unseen) == 3
        assert len(tracker) == 3

        # Next filter should return empty list
        second_unseen = tracker.filter_unseen(sample_jobs)
        assert len(second_unseen) == 0

    def test_save_and_load_persistence(self, tmp_path: Path, sample_jobs: list[Job]) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        storage_file = tmp_path / "subdir" / "seen_jobs.json"
        tracker = JobTracker(storage_path=storage_file, auto_save=True)

        tracker.mark_seen(sample_jobs[0])
        tracker.mark_seen(sample_jobs[1])

        assert storage_file.exists()

        # Load in a new tracker instance
        tracker2 = JobTracker(storage_path=storage_file)
        assert len(tracker2) == 2
        assert tracker2.is_seen(sample_jobs[0])
        assert tracker2.is_seen(sample_jobs[1])
        assert not tracker2.is_seen(sample_jobs[2])

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        storage_file = tmp_path / "nonexistent.json"
        tracker = JobTracker(storage_path=storage_file)
        assert len(tracker) == 0
        assert tracker.get_stats()["total_seen"] == 0

    def test_load_corrupt_file_graceful(self, tmp_path: Path) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        storage_file = tmp_path / "corrupt.json"
        storage_file.write_text("{invalid json content!@@#", encoding="utf-8")

        tracker = JobTracker(storage_path=storage_file)
        assert len(tracker) == 0  # Should handle gracefully without raising

    def test_clear(self, tmp_path: Path, sample_jobs: list[Job]) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        storage_file = tmp_path / "seen.json"
        tracker = JobTracker(storage_path=storage_file, auto_save=True)
        tracker.mark_seen(sample_jobs[0])
        tracker.mark_seen(sample_jobs[1])
        assert len(tracker) == 2

        tracker.clear()
        assert len(tracker) == 0
        assert not tracker.is_seen(sample_jobs[0])

        # Verify cleared file on disk
        tracker2 = JobTracker(storage_path=storage_file)
        assert len(tracker2) == 0

    def test_mark_many_seen(self, sample_jobs: list[Job]) -> None:
        from job_mcp.notifiers.tracker import JobTracker

        tracker = JobTracker()
        count = tracker.mark_many_seen([sample_jobs[0], sample_jobs[1], sample_jobs[0]])
        assert count == 2
        assert len(tracker) == 2


# ============================================================================
# Tests: TelegramNotifier Formatting & Splitting
# ============================================================================


class TestTelegramNotifierFormatting:
    """Tests for TelegramNotifier message formatting and chunking."""

    @pytest.fixture
    def rich_job(self) -> Job:
        return Job(
            job_id="job_full",
            title="Senior Full-Stack Engineer <Core>",
            company="Acme & Sons",
            location="Tel Aviv, Israel",
            work_mode=WorkMode.HYBRID,
            tech_stack=["Python", "FastAPI", "React", "PostgreSQL"],
            url="https://acme.example.com/jobs/123",
            apply_url="https://acme.example.com/apply/123",
            match_score=94.5,
            salary_range="35k - 42k ILS",
        )

    def test_format_single_job_html_escaping(self, rich_job: Job) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="fake_token", chat_id="12345", parse_mode="HTML")
        formatted = notifier.format_alert([rich_job], title="High Priority Matches")

        # Verify title & header
        assert "High Priority Matches" in formatted
        # Verify HTML escaping for <Core> and Acme & Sons
        assert "&lt;Core&gt;" in formatted
        assert "Acme &amp; Sons" in formatted
        assert "<Core>" not in formatted
        assert "Acme & Sons" not in formatted or "Acme &amp; Sons" in formatted

        # Verify fields present
        assert "Tel Aviv, Israel" in formatted
        assert "hybrid" in formatted.lower()
        assert "Python" in formatted
        assert "FastAPI" in formatted
        assert "94.5" in formatted or "95%" in formatted or "94%" in formatted
        assert "https://acme.example.com/jobs/123" in formatted or "https://acme.example.com/apply/123" in formatted

    def test_format_alert_empty_jobs(self) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="fake_token", chat_id="12345")
        formatted = notifier.format_alert([])
        assert "No new jobs" in formatted or "0 jobs" in formatted or formatted == ""

    def test_message_splitting_under_limit(self, rich_job: Job) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="fake_token", chat_id="12345", max_message_length=4096)
        text = notifier.format_alert([rich_job])
        chunks = notifier.split_message(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_message_splitting_over_limit(self) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="fake_token", chat_id="12345", max_message_length=150)
        jobs = [
            Job(
                job_id=f"job_{i}",
                title=f"Software Engineer Level {i}",
                company=f"Company {i}",
                url=f"https://example.com/job/{i}",
            )
            for i in range(10)
        ]
        text = notifier.format_alert(jobs)
        chunks = notifier.split_message(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 150


# ============================================================================
# Tests: TelegramNotifier HTTP Delivery & Health Check
# ============================================================================


class TestTelegramNotifierDelivery:
    """Tests for TelegramNotifier network operations, rate limiting, and error handling."""

    @pytest.fixture
    def test_jobs(self) -> list[Job]:
        return [
            Job(
                job_id="job_tg_1",
                title="AI Backend Developer",
                company="OpenCompany",
                location="Remote",
                url="https://example.com/ai-dev",
                match_score=88.0,
            )
        ]

    @pytest.mark.asyncio
    async def test_send_alert_success(self, test_jobs: list[Job]) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 101}}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        notifier = TelegramNotifier(
            bot_token="test_token",
            chat_id="test_chat_id",
            client=mock_client,
        )

        success = await notifier.send_alert(test_jobs, title="New Matches")
        assert success is True
        assert mock_client.post.call_count == 1

        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"]["chat_id"] == "test_chat_id"
        assert "New Matches" in call_kwargs["json"]["text"]
        assert call_kwargs["json"]["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_send_alert_empty_jobs(self) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        notifier = TelegramNotifier(
            bot_token="test_token",
            chat_id="test_chat_id",
            client=mock_client,
        )

        # Empty jobs list returns True without sending network requests
        success = await notifier.send_alert([])
        assert success is True
        assert mock_client.post.call_count == 0

    @pytest.mark.asyncio
    async def test_send_alert_unconfigured(self, test_jobs: list[Job]) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        with patch.dict("os.environ", {}, clear=True):
            notifier = TelegramNotifier(bot_token="", chat_id="")
            assert not notifier.is_configured
            success = await notifier.send_alert(test_jobs)
            assert success is False

    @pytest.mark.asyncio
    async def test_send_alert_multi_chunk(self) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_client.post.return_value = mock_response

        jobs = [
            Job(
                job_id=f"job_{i}",
                title=f"Engineer {i}",
                company="Company",
                url=f"https://example.com/{i}",
            )
            for i in range(15)
        ]

        # Force very small max length to trigger multi-chunk sending
        notifier = TelegramNotifier(
            bot_token="test_token",
            chat_id="test_chat",
            client=mock_client,
            max_message_length=200,
        )

        success = await notifier.send_alert(jobs)
        assert success is True
        assert mock_client.post.call_count > 1

    @pytest.mark.asyncio
    async def test_send_alert_http_error(self, test_jobs: list[Job]) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.text = "Bad Request: chat not found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_response
        )
        mock_client.post.return_value = mock_response

        notifier = TelegramNotifier(
            bot_token="test_token",
            chat_id="invalid_chat",
            client=mock_client,
        )

        success = await notifier.send_alert(test_jobs)
        assert success is False

    @pytest.mark.asyncio
    async def test_send_alert_network_timeout(self, test_jobs: list[Job]) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")

        notifier = TelegramNotifier(
            bot_token="test_token",
            chat_id="test_chat",
            client=mock_client,
        )

        success = await notifier.send_alert(test_jobs)
        assert success is False

    @pytest.mark.asyncio
    async def test_send_alert_rate_limiting_429(self, test_jobs: list[Job]) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_client = AsyncMock(spec=httpx.AsyncClient)

        rate_limit_resp = MagicMock(spec=httpx.Response)
        rate_limit_resp.status_code = 429
        rate_limit_resp.json.return_value = {
            "ok": False,
            "error_code": 429,
            "description": "Too Many Requests: retry after 1",
            "parameters": {"retry_after": 1},
        }

        success_resp = MagicMock(spec=httpx.Response)
        success_resp.status_code = 200
        success_resp.json.return_value = {"ok": True}

        mock_client.post.side_effect = [rate_limit_resp, success_resp]

        notifier = TelegramNotifier(
            bot_token="test_token",
            chat_id="test_chat",
            client=mock_client,
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            success = await notifier.send_alert(test_jobs)
            assert success is True
            assert mock_client.post.call_count == 2
            mock_sleep.assert_awaited_once_with(1)

    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"id": 123456, "is_bot": True, "first_name": "JobBot"},
        }
        mock_client.get.return_value = mock_resp

        notifier = TelegramNotifier(bot_token="valid_token", chat_id="123", client=mock_client)
        assert await notifier.check_health() is True
        mock_client.get.assert_called_once_with(
            "https://api.telegram.org/botvalid_token/getMe",
            timeout=10.0,
        )

    @pytest.mark.asyncio
    async def test_check_health_failure(self) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"ok": False, "error_code": 401, "description": "Unauthorized"}
        mock_client.get.return_value = mock_resp

        notifier = TelegramNotifier(bot_token="invalid_token", chat_id="123", client=mock_client)
        assert await notifier.check_health() is False

    @pytest.mark.asyncio
    async def test_check_health_unconfigured(self) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="", chat_id="")
        assert await notifier.check_health() is False

    def test_env_var_initialization(self) -> None:
        from job_mcp.notifiers.telegram import TelegramNotifier

        with patch.dict(
            "os.environ",
            {"TELEGRAM_BOT_TOKEN": "env_token_999", "TELEGRAM_CHAT_ID": "env_chat_888"},
        ):
            notifier = TelegramNotifier()
            assert notifier.bot_token == "env_token_999"
            assert notifier.chat_id == "env_chat_888"
            assert notifier.is_configured is True


# ============================================================================
# Tests: Module Exports
# ============================================================================


def test_notifiers_package_exports() -> None:
    from job_mcp.notifiers import BaseNotifier, JobTracker, TelegramNotifier

    assert BaseNotifier is not None
    assert TelegramNotifier is not None
    assert JobTracker is not None
