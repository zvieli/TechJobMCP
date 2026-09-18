"""Unit tests for job source base contracts, metadata, timeouts, and overrides."""

from __future__ import annotations

import pytest


def test_base_source_get_timeout_default_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from job_mcp.sources.base import BaseJobSource

    class DummySource(BaseJobSource):
        source_id = "dummy_src"
        display_name = "Dummy"
        timeout = 14.5

        async def fetch_jobs(self, preferences=None, limit=50):
            return []

    src = DummySource()
    assert src.get_timeout() == 14.5
    assert src.get_metadata().default_timeout == 14.5

    # Test environment variable override: SOURCE_TIMEOUT_DUMMY_SRC
    monkeypatch.setenv("SOURCE_TIMEOUT_DUMMY_SRC", "22.0")
    assert src.get_timeout() == 22.0
    assert src.get_metadata().default_timeout == 22.0
