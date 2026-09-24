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


def test_base_source_get_timeout_validates_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    from job_mcp.sources.base import BaseJobSource

    class DummySource(BaseJobSource):
        source_id = "dummy_src"
        display_name = "Dummy"
        timeout = -5.0

        async def fetch_jobs(self, preferences=None, limit=50):
            return []

    src = DummySource()
    # Negative timeout falls back to default 15.0
    assert src.get_timeout() == 15.0

    # Non-positive or invalid env var overrides are ignored
    monkeypatch.setenv("SOURCE_TIMEOUT_DUMMY_SRC", "-10.0")
    assert src.get_timeout() == 15.0

    monkeypatch.setenv("SOURCE_TIMEOUT_DUMMY_SRC", "0")
    assert src.get_timeout() == 15.0

    monkeypatch.setenv("SOURCE_TIMEOUT_DUMMY_SRC", "not_a_number")
    assert src.get_timeout() == 15.0


def test_registered_sources_have_tailored_timeouts(monkeypatch):
    import os
    for k in list(os.environ.keys()):
        if k.startswith("SOURCE_TIMEOUT_"):
            monkeypatch.delenv(k, raising=False)

    from job_mcp.sources.registry import create_default_registry

    reg = create_default_registry()
    sources = {s.source_id: s.get_timeout() for s in reg.get_all()}

    expected_timeouts = {
        "eightfold": 10.0,
        "hiremetech": 12.0,
        "direct_tech": 15.0,
        "workday": 15.0,
        "lever": 15.0,
        "comeet": 18.0,
        "jobify": 18.0,
        "greenhouse": 20.0,
        "linkedin": 20.0,
        "gotfriends": 25.0,
    }
    for sid, expected_t in expected_timeouts.items():
        if sid in sources:
            assert sources[sid] == expected_t, f"Source {sid} timeout mismatch: {sources[sid]} != {expected_t}"


def test_all_11_sources_have_tailored_timeouts():
    from job_mcp.sources.authenticated.hiremetech import HireMeTechSource
    from job_mcp.sources.authenticated.linkedin import LinkedInSource
    from job_mcp.sources.enterprise.direct_tech import DirectTechSource
    from job_mcp.sources.enterprise.workday import WorkdaySource
    from job_mcp.sources.public.alljobs import AllJobsSource
    from job_mcp.sources.public.comeet import ComeetSource
    from job_mcp.sources.public.eightfold import EightfoldAISource
    from job_mcp.sources.public.gotfriends import GotFriendsSource
    from job_mcp.sources.public.greenhouse import GreenhouseSource
    from job_mcp.sources.public.jobify import JobifySource
    from job_mcp.sources.public.lever import LeverSource

    assert HireMeTechSource.timeout == 12.0
    assert LinkedInSource.timeout == 20.0
    assert DirectTechSource.timeout == 15.0
    assert WorkdaySource.timeout == 15.0
    assert AllJobsSource.timeout == 10.0
    assert ComeetSource.timeout == 18.0
    assert EightfoldAISource.timeout == 10.0
    assert GotFriendsSource.timeout == 25.0
    assert GreenhouseSource.timeout == 20.0
    assert JobifySource.timeout == 18.0
    assert LeverSource.timeout == 15.0

    # Ensure AllJobs instance get_timeout returns 10.0 while preserving request timeout
    alljobs = AllJobsSource()
    assert alljobs.get_timeout() == 10.0
    assert alljobs.timeout == 2.0

