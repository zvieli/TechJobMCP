"""Unit and integration tests for JobAggregator and FastMCP multi-source tools."""

from __future__ import annotations

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from job_mcp.core.api_client import JobCache
from job_mcp.core.auth import SessionManager
from job_mcp.main import get_job_matches, list_job_sources
from job_mcp.models.schemas import Job, JobPreferences, WorkMode
from job_mcp.sources import (
    BaseJobSource,
    ComeetCompany,
    ComeetSource,
    HireMeTechSource,
    JobAggregator,
    SourceMetadata,
    SourceRegistry,
    create_default_registry,
)
from job_mcp.sources.aggregator import DEFAULT_SOURCE_TIMEOUT


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
        delay: float = 0.0,
    ) -> None:
        self.source_id = source_id
        self.display_name = display_name or source_id.capitalize()
        self.description = f"Mock {self.display_name} source"
        self._jobs = jobs or []
        self._is_healthy = is_healthy
        self._raise_on_fetch = raise_on_fetch
        self._raise_on_health = raise_on_health
        self._delay = delay
        self.fetch_call_count = 0
        self.health_call_count = 0

    async def fetch_jobs(
        self,
        preferences: JobPreferences | None = None,
        limit: int = 50,
    ) -> list[Job]:
        self.fetch_call_count += 1
        if self._delay > 0:
            import asyncio
            await asyncio.sleep(self._delay)
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

    async def test_aggregator_source_timeout_configuration(self) -> None:
        """Verify source_timeout defaults to DEFAULT_SOURCE_TIMEOUT and can be overridden."""
        agg_default = JobAggregator()
        self.assertEqual(agg_default.source_timeout, DEFAULT_SOURCE_TIMEOUT)

        agg_custom = JobAggregator(source_timeout=1.5)
        self.assertEqual(agg_custom.source_timeout, 1.5)

    async def test_aggregator_per_source_timeout_cutoff(self) -> None:
        """Verify slow source times out without blocking or failing fast sources."""
        slow_source = MockSource("slow_source", jobs=[self.job_hmt], delay=5.0)
        fast_source = MockSource(
            "fast_source",
            jobs=[self.job_hmt, self.job_alljobs],
            delay=0.0,
        )

        reg = SourceRegistry()
        reg.register(slow_source)
        reg.register(fast_source)

        agg = JobAggregator(registry=reg, source_timeout=0.5)

        start_time = time.monotonic()
        jobs = await agg.fetch_all_jobs(force_refresh=True)
        elapsed = time.monotonic() - start_time

        # Ensure execution cutoff happened fast (< 1.5s, well below the 5.0s slow source delay)
        self.assertLess(elapsed, 1.5)
        # Fast source jobs should have been returned successfully
        self.assertEqual(len(jobs), 2)
        self.assertEqual(fast_source.fetch_call_count, 1)
        self.assertEqual(slow_source.fetch_call_count, 1)

    async def test_fetch_all_jobs_partial_source_failure_graceful_degradation(self) -> None:
        """Verify partial source failure with asyncio.gather(return_exceptions=True) returns jobs from successful sources."""
        src_success = MockSource("good_source", jobs=[self.job_hmt, self.job_alljobs])
        src_error = MockSource("failing_source", raise_on_fetch=RuntimeError("Connection reset by peer"))
        src_timeout = MockSource("timeout_source", delay=5.0)

        reg = SourceRegistry()
        reg.register(src_success)
        reg.register(src_error)
        reg.register(src_timeout)

        agg = JobAggregator(registry=reg, source_timeout=0.3)
        jobs = await agg.fetch_all_jobs(force_refresh=True)

        # Successful source returns its jobs despite the failure and timeout of others
        self.assertEqual(len(jobs), 2)
        self.assertEqual({j.job_id for j in jobs}, {"hmt-1", "aj-1"})
        self.assertEqual(src_success.fetch_call_count, 1)
        self.assertEqual(src_error.fetch_call_count, 1)
        self.assertEqual(src_timeout.fetch_call_count, 1)

    async def test_fetch_all_jobs_gather_exception_result_handling(self) -> None:
        """Verify that if asyncio.gather returns an Exception object directly, it is caught and logged gracefully."""
        src1 = MockSource("src1", jobs=[self.job_hmt])
        src2 = MockSource("src2", jobs=[self.job_alljobs])

        reg = SourceRegistry()
        reg.register(src1)
        reg.register(src2)

        agg = JobAggregator(registry=reg)

        exc = RuntimeError("Simulated unhandled task crash")

        async def mock_gather(*coros, **kwargs):
            for c in coros:
                c.close()
            return [[self.job_hmt], exc]

        with patch("job_mcp.sources.aggregator.asyncio.gather", side_effect=mock_gather):
            with patch("job_mcp.sources.aggregator.logger.warning") as mock_warning:
                jobs = await agg.fetch_all_jobs(force_refresh=True)

                self.assertEqual(len(jobs), 1)
                self.assertEqual(jobs[0].job_id, "hmt-1")
                # Verify warning logged with structured format
                mock_warning.assert_called_with("Source '%s' fetch failed: %s", "src2", exc)


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

    async def test_all_seven_sources_aggregation_and_deduplication(self) -> None:
        """Test aggregation and deduplication across all 7 supported sources simultaneously."""
        job_hmt = Job(job_id="hmt_1", title="Staff Engineer", company="GlobalCorp", tech_stack=["Python"], source="hiremetech", sources=["hiremetech"])
        job_comeet = Job(job_id="cmt_1", title="Frontend Architect", company="WebCo", tech_stack=["React"], source="comeet", sources=["comeet"])
        job_aj = Job(job_id="aj_1", title="Data Engineer", company="DataInc", tech_stack=["SQL"], source="alljobs", sources=["alljobs"])
        # Workday job duplicates HireMeTech job
        job_wd = Job(job_id="wd_1", title="Staff Engineer", company="GlobalCorp", tech_stack=["AWS", "Docker"], source="workday", sources=["workday"])
        job_eightfold = Job(job_id="ef_1", title="AI Researcher", company="AILab", tech_stack=["PyTorch"], source="eightfold", sources=["eightfold"])
        job_direct = Job(job_id="dt_1", title="Cloud Architect", company="Google", tech_stack=["GCP", "Go"], source="direct_tech", sources=["direct_tech"])
        # LinkedIn job duplicates Comeet job
        job_li = Job(job_id="li_1", title="Frontend Architect", company="WebCo", tech_stack=["TypeScript", "Next.js"], source="linkedin", sources=["linkedin"])

        reg = SourceRegistry()
        reg.register(MockSource("hiremetech", jobs=[job_hmt]))
        reg.register(MockSource("comeet", jobs=[job_comeet]))
        reg.register(MockSource("alljobs", jobs=[job_aj]))
        reg.register(MockSource("workday", jobs=[job_wd]))
        reg.register(MockSource("eightfold", jobs=[job_eightfold]))
        reg.register(MockSource("direct_tech", jobs=[job_direct]))
        reg.register(MockSource("linkedin", jobs=[job_li]))

        agg = JobAggregator(registry=reg)
        jobs = await agg.fetch_all_jobs(force_refresh=True)

        # 7 jobs across 7 sources -> 5 unique jobs after deduplication
        self.assertEqual(len(jobs), 5)

        # Verify merged Staff Engineer job
        staff_job = next(j for j in jobs if j.title == "Staff Engineer")
        self.assertEqual(staff_job.company, "GlobalCorp")
        self.assertIn("hiremetech", staff_job.sources)
        self.assertIn("workday", staff_job.sources)
        self.assertIn("Python", staff_job.tech_stack)
        self.assertIn("AWS", staff_job.tech_stack)

        # Verify merged Frontend Architect job
        fe_job = next(j for j in jobs if j.title == "Frontend Architect")
        self.assertEqual(fe_job.company, "WebCo")
        self.assertIn("comeet", fe_job.sources)
        self.assertIn("linkedin", fe_job.sources)
        self.assertIn("React", fe_job.tech_stack)
        self.assertIn("TypeScript", fe_job.tech_stack)

    async def test_all_seven_sources_health_checks(self) -> None:
        """Test health checking across all 7 sources concurrently."""
        reg = SourceRegistry()
        reg.register(MockSource("hiremetech", is_healthy=True))
        reg.register(MockSource("comeet", is_healthy=True))
        reg.register(MockSource("alljobs", is_healthy=False))
        reg.register(MockSource("workday", is_healthy=True))
        reg.register(MockSource("eightfold", is_healthy=False))
        reg.register(MockSource("direct_tech", is_healthy=True))
        reg.register(MockSource("linkedin", raise_on_health=RuntimeError("LinkedIn rate limit")))

        agg = JobAggregator(registry=reg)
        health_map = await agg.check_all_health()

        self.assertEqual(len(health_map), 7)
        self.assertTrue(health_map["hiremetech"])
        self.assertTrue(health_map["comeet"])
        self.assertFalse(health_map["alljobs"])
        self.assertTrue(health_map["workday"])
        self.assertFalse(health_map["eightfold"])
        self.assertTrue(health_map["direct_tech"])
        self.assertFalse(health_map["linkedin"])

    def test_create_default_registry_all_flags(self) -> None:
        """Test create_default_registry supports enabling all 7 sources."""
        # 1. Default (only HireMeTech + Comeet)
        reg_default = create_default_registry(
            enable_alljobs=False,
            enable_workday=False,
            enable_eightfold=False,
            enable_direct_tech=False,
            enable_linkedin=False,
        )
        self.assertEqual(len(reg_default), 2)
        self.assertIn("hiremetech", reg_default)
        self.assertIn("comeet", reg_default)

        # 2. All 7 enabled
        reg_all = create_default_registry(
            enable_alljobs=True,
            enable_workday=True,
            enable_eightfold=True,
            enable_direct_tech=True,
            enable_linkedin=True,
        )
        self.assertEqual(len(reg_all), 7)
        self.assertIn("hiremetech", reg_all)
        self.assertIn("comeet", reg_all)
        self.assertIn("alljobs", reg_all)
        self.assertIn("workday", reg_all)
        self.assertIn("eightfold", reg_all)
        self.assertIn("direct_tech", reg_all)
        self.assertIn("linkedin", reg_all)


if __name__ == "__main__":
    unittest.main()

