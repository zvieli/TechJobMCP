"""Unit and integration tests for JobAggregator and FastMCP multi-source tools."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from hireme_mcp.core.api_client import JobCache
from hireme_mcp.core.auth import SessionManager
from hireme_mcp.main import get_job_matches, list_job_sources
from hireme_mcp.models.schemas import Job, JobPreferences, WorkMode
from hireme_mcp.sources import (
    BaseJobSource,
    ComeetCompany,
    ComeetSource,
    HireMeTechSource,
    JobAggregator,
    SourceMetadata,
    SourceRegistry,
    create_default_registry,
)


class MockSource(BaseJobSource):
    """Mock job source for testing JobAggregator."""

    def __init__(
        self,
        source_id: str,
        display_name: str = "",
        jobs: list[Job] | None = None,
        is_healthy: bool = True,
        raise_on_fetch: Exception | None = None,
        raise_on_health: Exception | None = None,
    ) -> None:
        self.source_id = source_id
        self.display_name = display_name or source_id.capitalize()
        self.description = f"Mock {self.display_name} source"
        self._jobs = jobs or []
        self._is_healthy = is_healthy
        self._raise_on_fetch = raise_on_fetch
        self._raise_on_health = raise_on_health
        self.fetch_call_count = 0
        self.health_call_count = 0

    async def fetch_jobs(
        self,
        preferences: JobPreferences | None = None,
        limit: int = 50,
    ) -> list[Job]:
        self.fetch_call_count += 1
        if self._raise_on_fetch:
            raise self._raise_on_fetch
        return self._jobs[:limit]

    async def check_health(self) -> bool:
        self.health_call_count += 1
        if self._raise_on_health:
            raise self._raise_on_health
        return self._is_healthy


class TestJobAggregator(unittest.IsolatedAsyncioTestCase):
    """Tests for JobAggregator concurrent fetching, deduplication, and health checking."""

    def setUp(self) -> None:
        """Set up test fixtures with sample jobs and sources."""
        self.job_hmt = Job(
            job_id="hmt-1",
            title="Senior Python Engineer",
            company="Acme Corp",
            location="Tel Aviv",
            work_mode=WorkMode.HYBRID,
            tech_stack=["Python", "FastAPI", "PostgreSQL"],
            description="Building scalable backend APIs",
            source="hiremetech",
            sources=["hiremetech"],
            link="https://hiremetech.com/jobs/1",
        )
        self.job_comeet = Job(
            job_id="comeet-1",
            title="Senior Python Engineer",  # duplicate of job_hmt
            company="Acme Corp",
            location="Tel Aviv",
            work_mode=WorkMode.HYBRID,
            tech_stack=["Python", "Django", "Docker"],
            description="We are hiring a Python developer",
            source="comeet",
            sources=["comeet"],
            link="https://www.comeet.co/jobs/acme/1",
        )
        self.job_alljobs = Job(
            job_id="aj-1",
            title="Frontend React Developer",
            company="Frontend Stars",
            location="Remote",
            work_mode=WorkMode.REMOTE,
            tech_stack=["React", "TypeScript", "Next.js"],
            description="React frontend development",
            source="alljobs",
            sources=["alljobs"],
            link="https://www.alljobs.co.il/jobs/100",
        )

    async def test_aggregator_initialization_defaults(self) -> None:
        """Verify JobAggregator initializes with global registry when registry=None."""
        agg = JobAggregator()
        self.assertIsNotNone(agg.registry)
        self.assertIsInstance(agg.registry, SourceRegistry)
        self.assertIsNone(agg.cache)

    async def test_fetch_all_jobs_concurrent_execution(self) -> None:
        """Verify fetch_all_jobs queries all registered sources concurrently."""
        src1 = MockSource("src1", jobs=[self.job_hmt])
        src2 = MockSource("src2", jobs=[self.job_alljobs])

        reg = SourceRegistry()
        reg.register(src1)
        reg.register(src2)

        agg = JobAggregator(registry=reg)
        jobs = await agg.fetch_all_jobs(force_refresh=True)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(src1.fetch_call_count, 1)
        self.assertEqual(src2.fetch_call_count, 1)

    async def test_fetch_all_jobs_deduplication_and_source_merging(self) -> None:
        """Verify fetch_all_jobs deduplicates identical jobs across sources and merges metadata."""
        src_hmt = MockSource("hiremetech", jobs=[self.job_hmt])
        src_comeet = MockSource("comeet", jobs=[self.job_comeet])
        src_aj = MockSource("alljobs", jobs=[self.job_alljobs])

        reg = SourceRegistry()
        reg.register(src_hmt)
        reg.register(src_comeet)
        reg.register(src_aj)

        agg = JobAggregator(registry=reg)
        jobs = await agg.fetch_all_jobs(force_refresh=True)

        # 3 jobs fetched -> 2 unique jobs after deduplication
        self.assertEqual(len(jobs), 2)

        # Find deduplicated Python job
        py_job = next(j for j in jobs if "Python" in j.title)
        self.assertIn("hiremetech", py_job.sources)
        self.assertIn("comeet", py_job.sources)
        # Tech stack should be combined: Python, FastAPI, PostgreSQL, Django, Docker
        self.assertIn("FastAPI", py_job.tech_stack)
        self.assertIn("Django", py_job.tech_stack)

    async def test_fetch_all_jobs_error_isolation(self) -> None:
        """Verify one failing source does not crash aggregation or prevent other sources from returning."""
        src_good = MockSource("good_source", jobs=[self.job_hmt])
        src_bad = MockSource("bad_source", raise_on_fetch=RuntimeError("Network failure 500"))

        reg = SourceRegistry()
        reg.register(src_good)
        reg.register(src_bad)

        agg = JobAggregator(registry=reg)
        jobs = await agg.fetch_all_jobs(force_refresh=True)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, "hmt-1")
        self.assertEqual(src_good.fetch_call_count, 1)
        self.assertEqual(src_bad.fetch_call_count, 1)

    async def test_fetch_all_jobs_with_preferences_scoring(self) -> None:
        """Verify passing JobPreferences ranks and scores jobs according to criteria."""
        src_hmt = MockSource("hiremetech", jobs=[self.job_hmt])
        src_aj = MockSource("alljobs", jobs=[self.job_alljobs])

        reg = SourceRegistry()
        reg.register(src_hmt)
        reg.register(src_aj)

        agg = JobAggregator(registry=reg)
        prefs = JobPreferences(
            tech_stack=["Python", "FastAPI"],
            work_mode=WorkMode.HYBRID,
        )
        jobs = await agg.fetch_all_jobs(preferences=prefs, force_refresh=True)

        self.assertGreater(len(jobs), 0)
        top_job = jobs[0]
        self.assertEqual(top_job.job_id, "hmt-1")
        self.assertGreater(top_job.match_score, 70.0)

    async def test_fetch_all_jobs_cache_hit_and_force_refresh(self) -> None:
        """Verify cache hit returns cached jobs without calling sources, and force_refresh bypasses cache."""
        cache = JobCache(ttl_minutes=10)
        cache.update([self.job_hmt, self.job_alljobs])

        src = MockSource("hiremetech", jobs=[self.job_hmt])
        reg = SourceRegistry()
        reg.register(src)

        agg = JobAggregator(registry=reg, cache=cache)

        # 1. Cache hit: source.fetch_jobs not called
        cached_res = await agg.fetch_all_jobs(force_refresh=False)
        self.assertEqual(len(cached_res), 2)
        self.assertEqual(src.fetch_call_count, 0)

        # 2. Force refresh: source.fetch_jobs is called
        refreshed_res = await agg.fetch_all_jobs(force_refresh=True)
        self.assertEqual(len(refreshed_res), 1)
        self.assertEqual(src.fetch_call_count, 1)

    async def test_fetch_all_jobs_filter_by_sources(self) -> None:
        """Verify sources parameter only queries specified sources."""
        src1 = MockSource("hiremetech", jobs=[self.job_hmt])
        src2 = MockSource("alljobs", jobs=[self.job_alljobs])

        reg = SourceRegistry()
        reg.register(src1)
        reg.register(src2)

        agg = JobAggregator(registry=reg)
        jobs = await agg.fetch_all_jobs(sources=["alljobs"], force_refresh=True)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, "aj-1")
        self.assertEqual(src1.fetch_call_count, 0)
        self.assertEqual(src2.fetch_call_count, 1)

    async def test_fetch_all_jobs_empty_active_sources(self) -> None:
        """Verify querying non-existent source returns empty list gracefully."""
        reg = SourceRegistry()
        reg.register(MockSource("src1", jobs=[self.job_hmt]))

        agg = JobAggregator(registry=reg)
        jobs = await agg.fetch_all_jobs(sources=["non_existent_source"], force_refresh=True)
        self.assertEqual(jobs, [])

    async def test_check_all_health(self) -> None:
        """Verify check_all_health checks health of all registered sources concurrently."""
        src1 = MockSource("src1", is_healthy=True)
        src2 = MockSource("src2", is_healthy=False)
        src3 = MockSource("src3", raise_on_health=RuntimeError("Health check connection timeout"))

        reg = SourceRegistry()
        reg.register(src1)
        reg.register(src2)
        reg.register(src3)

        agg = JobAggregator(registry=reg)
        health_map = await agg.check_all_health()

        self.assertEqual(health_map, {
            "src1": True,
            "src2": False,
            "src3": False,
        })
        self.assertEqual(src1.health_call_count, 1)
        self.assertEqual(src2.health_call_count, 1)
        self.assertEqual(src3.health_call_count, 1)

    async def test_check_all_health_empty_registry(self) -> None:
        """Verify check_all_health returns empty dict when no sources are registered."""
        reg = SourceRegistry()
        agg = JobAggregator(registry=reg)
        health_map = await agg.check_all_health()
        self.assertEqual(health_map, {})


class TestMultiSourceMcpTools(unittest.IsolatedAsyncioTestCase):
    """Tests for list_job_sources and get_job_matches FastMCP tools."""

    def setUp(self) -> None:
        """Set up test environment and mock jobs."""
        self.job1 = Job(
            job_id="hmt-100",
            title="DevOps Engineer",
            company="CloudTech",
            tech_stack=["AWS", "Terraform", "Docker"],
            source="hiremetech",
            sources=["hiremetech"],
        )
        self.job2 = Job(
            job_id="comeet-200",
            title="Data Scientist",
            company="AI Labs",
            tech_stack=["Python", "PyTorch"],
            source="comeet",
            sources=["comeet"],
        )

    def _create_mock_context(
        self,
        registry: SourceRegistry,
        aggregator: JobAggregator,
        cache: JobCache,
        session_mgr: SessionManager | None = None,
    ) -> MagicMock:
        """Create mock FastMCP Context with full lifespan state."""
        ctx = MagicMock(spec=Context)
        ctx.lifespan_context = {
            "registry": registry,
            "aggregator": aggregator,
            "cache": cache,
            "session": session_mgr or MagicMock(spec=SessionManager),
        }
        return ctx

    async def test_list_job_sources_tool(self) -> None:
        """Test list_job_sources returns source metadata list and health status."""
        src1 = MockSource("hiremetech", display_name="HireMeTech", is_healthy=True)
        src2 = MockSource("comeet", display_name="Comeet", is_healthy=True)
        src3 = MockSource("alljobs", display_name="AllJobs", is_healthy=False)

        reg = SourceRegistry()
        reg.register(src1)
        reg.register(src2)
        reg.register(src3)

        cache = JobCache()
        agg = JobAggregator(registry=reg, cache=cache)
        ctx = self._create_mock_context(reg, agg, cache)

        res = await list_job_sources(ctx=ctx)

        self.assertTrue(res["success"])
        self.assertIn("Retrieved 3 registered", res["message"])
        self.assertIn("trace_id", res)
        self.assertIsNotNone(res["trace_id"])

        data = res["data"]
        self.assertIn("sources", data)
        self.assertIn("health", data)
        self.assertEqual(len(data["sources"]), 3)
        self.assertEqual(data["health"]["hiremetech"], True)
        self.assertEqual(data["health"]["comeet"], True)
        self.assertEqual(data["health"]["alljobs"], False)

    async def test_get_job_matches_with_sources_filter(self) -> None:
        """Test get_job_matches with sources=['comeet'] only fetches from Comeet."""
        src_hmt = MockSource("hiremetech", jobs=[self.job1])
        src_comeet = MockSource("comeet", jobs=[self.job2])

        reg = SourceRegistry()
        reg.register(src_hmt)
        reg.register(src_comeet)

        cache = JobCache()
        agg = JobAggregator(registry=reg, cache=cache)
        ctx = self._create_mock_context(reg, agg, cache)

        res = await get_job_matches(sources=["comeet"], force_refresh=True, ctx=ctx)

        self.assertTrue(res["success"])
        self.assertIn("Successfully fetched 1 live", res["message"])
        self.assertEqual(len(res["data"]), 1)
        self.assertEqual(res["data"][0]["job_id"], "comeet-200")
        self.assertEqual(res["data"][0]["source"], "comeet")
        self.assertEqual(src_hmt.fetch_call_count, 0)
        self.assertEqual(src_comeet.fetch_call_count, 1)

    async def test_get_job_matches_all_sources_aggregation(self) -> None:
        """Test get_job_matches without sources parameter queries all active sources."""
        src_hmt = MockSource("hiremetech", jobs=[self.job1])
        src_comeet = MockSource("comeet", jobs=[self.job2])

        reg = SourceRegistry()
        reg.register(src_hmt)
        reg.register(src_comeet)

        cache = JobCache()
        agg = JobAggregator(registry=reg, cache=cache)
        ctx = self._create_mock_context(reg, agg, cache)

        res = await get_job_matches(sources=None, force_refresh=True, ctx=ctx)

        self.assertTrue(res["success"])
        self.assertEqual(len(res["data"]), 2)
        self.assertEqual(src_hmt.fetch_call_count, 1)
        self.assertEqual(src_comeet.fetch_call_count, 1)


if __name__ == "__main__":
    unittest.main()
