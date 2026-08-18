"""Tests for browser session resilience and recovery."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from job_mcp.core.auth import SessionManager


class TestSessionManagerRecovery:
    @pytest.fixture
    def session(self, tmp_path):
        return SessionManager(user_data_dir=tmp_path / "profile", headless=True)

    @pytest.mark.asyncio
    async def test_recover_calls_shutdown_then_initialize(self, session):
        """Recovery should cleanly teardown then re-initialize."""
        session._initialized = True
        session.shutdown = AsyncMock()
        session.initialize = AsyncMock()
        session.check_session_health = AsyncMock(return_value=True)

        result = await session.recover()

        session.shutdown.assert_awaited_once()
        session.initialize.assert_awaited_once()
        session.check_session_health.assert_awaited_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_recover_returns_false_on_init_failure(self, session):
        """If re-initialize fails, recover returns False."""
        session._initialized = True
        session.shutdown = AsyncMock()
        session.initialize = AsyncMock(side_effect=RuntimeError("chromium crashed"))
        session.check_session_health = AsyncMock(return_value=False)

        result = await session.recover()

        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_ready_success_first_try(self, session):
        """ensure_ready returns page on first successful attempt."""
        mock_page = MagicMock()
        session.initialize = AsyncMock()
        session.check_session_health = AsyncMock(return_value=True)
        session.get_page = AsyncMock(return_value=mock_page)

        page = await session.ensure_ready(max_retries=3)

        assert page is mock_page
        session.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ensure_ready_retries_on_unhealthy(self, session):
        """ensure_ready retries recovery when health check fails."""
        mock_page = MagicMock()
        session.initialize = AsyncMock()
        # First call: unhealthy, second call after recovery: healthy
        session.check_session_health = AsyncMock(side_effect=[False, True])
        session.recover = AsyncMock(return_value=True)
        session.get_page = AsyncMock(return_value=mock_page)

        page = await session.ensure_ready(max_retries=3)

        assert page is mock_page
        session.recover.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ensure_ready_retries_on_init_exception(self, session):
        """ensure_ready retries when initialize() throws."""
        mock_page = MagicMock()
        # First init fails, second succeeds
        session.initialize = AsyncMock(
            side_effect=[RuntimeError("browser crashed"), None]
        )
        session.check_session_health = AsyncMock(return_value=True)
        session.get_page = AsyncMock(return_value=mock_page)

        page = await session.ensure_ready(max_retries=3)

        assert page is mock_page
        assert session.initialize.await_count == 2

    @pytest.mark.asyncio
    async def test_ensure_ready_exhausts_retries_raises(self, session):
        """ensure_ready raises RuntimeError after exhausting retries."""
        session.initialize = AsyncMock(
            side_effect=RuntimeError("always fails")
        )

        with pytest.raises(RuntimeError, match="after 2 attempts"):
            await session.ensure_ready(max_retries=2)


class TestLifespanRecovery:
    @pytest.mark.asyncio
    async def test_lifespan_logs_warning_on_init_failure(self):
        """Lifespan should not crash if initial browser launch fails."""
        from job_mcp.main import browser_lifespan, mcp

        with patch.object(SessionManager, "initialize", new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = RuntimeError("no display")
            with patch.object(SessionManager, "shutdown", new_callable=AsyncMock):
                async with browser_lifespan(mcp) as ctx:
                    assert "session" in ctx
                    # Session exists but might not be initialized
                    assert isinstance(ctx["session"], SessionManager)
