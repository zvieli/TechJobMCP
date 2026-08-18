"""Multi-source job aggregator coordinating concurrent fetching, deduplication, and ranking."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Optional

from job_mcp.core.api_client import JobCache, filter_jobs
from job_mcp.models.schemas import Job, JobPreferences
from job_mcp.sources.base import BaseJobSource
from job_mcp.sources.dedup import deduplicate_jobs
from job_mcp.utils.logger import get_logger

if TYPE_CHECKING:
    from job_mcp.sources import SourceRegistry

logger = get_logger(__name__)


DEFAULT_SOURCE_TIMEOUT: float = 3.0


class JobAggregator:
    """Aggregates and deduplicates job listings across multiple registered sources."""

    def __init__(
        self,
        registry: Optional[SourceRegistry] = None,
        cache: Optional[JobCache] = None,
        source_timeout: float = DEFAULT_SOURCE_TIMEOUT,
    ) -> None:
        """Initialize JobAggregator.

        Args:
            registry: Optional SourceRegistry instance. If None, uses default registry.
            cache: Optional JobCache instance for storing aggregated jobs.
            source_timeout: Maximum timeout in seconds allowed per source fetch.
        """
        if registry is None:
            from job_mcp.sources import registry as default_registry

            self.registry = default_registry
        else:
            self.registry = registry

        self.cache = cache
        self.source_timeout = source_timeout

    async def fetch_all_jobs(
        self,
        sources: Optional[list[str]] = None,
        preferences: Optional[JobPreferences] = None,
        limit_per_source: int = 50,
        force_refresh: bool = False,
    ) -> list[Job]:
        """Fetch job listings concurrently from active registered sources, deduplicate, and score.

        Args:
            sources: Optional list of source IDs to query (e.g. ['hiremetech', 'comeet', 'alljobs']).
                     If None, fetches from all active registered sources.
            preferences: Optional JobPreferences for filtering, keyword matching, and match scoring.
            limit_per_source: Maximum number of jobs to fetch per individual source.
            force_refresh: If True, bypasses cache and forces live fetching across sources.

        Returns:
            list[Job]: Unified, deduplicated, and optionally scored list of Job listings.
        """
        # 1. Check cache if fresh and not forcing refresh
        if not force_refresh and self.cache is not None and not self.cache.is_stale:
            cached_jobs = self.cache.get_all()
            if cached_jobs:
                logger.info("Returning %d jobs from cache in JobAggregator.", len(cached_jobs))
                results = cached_jobs
                if sources is not None:
                    source_set = set(sources)
                    results = [
                        j
                        for j in results
                        if j.source in source_set or any(s in source_set for s in getattr(j, "sources", []))
                    ]
                if preferences is not None:
                    results = filter_jobs(results, preferences)
                return results

        # 2. Get active sources
        active_sources = self.registry.get_active(sources)
        if not active_sources:
            logger.warning("No active sources found in registry for filter: %s", sources)
            return []

        # 3. Fetch from all sources concurrently with per-source timeout & error isolation
        logger.info(
            "Fetching jobs concurrently from %d source(s): %s (timeout: %.1fs)",
            len(active_sources),
            [s.source_id for s in active_sources],
            self.source_timeout,
        )

        async def _fetch_with_timeout(src: BaseJobSource) -> list[Job]:
            t0 = time.perf_counter()
            try:
                jobs = await asyncio.wait_for(
                    src.fetch_jobs(preferences=preferences, limit=limit_per_source),
                    timeout=self.source_timeout,
                )
                duration_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    "Source fetch completed",
                    source_id=src.source_id,
                    jobs_count=len(jobs),
                    duration_ms=round(duration_ms, 2),
                )
                return jobs
            except asyncio.TimeoutError:
                logger.warning(
                    "Source '%s' timed out after %.1fs.",
                    src.source_id,
                    self.source_timeout,
                )
                return []
            except Exception as exc:
                logger.warning("Source '%s' fetch failed: %s", src.source_id, exc)
                return []

        results = await asyncio.gather(
            *[_fetch_with_timeout(s) for s in active_sources]
        )

        all_jobs: list[Job] = []
        for source, res in zip(active_sources, results):
            if res:
                logger.info("Source '%s' returned %d jobs.", source.source_id, len(res))
                all_jobs.extend(res)

        # 4. Deduplicate across sources
        deduped = deduplicate_jobs(all_jobs)
        logger.info(
            "Aggregated and deduplicated %d jobs down to %d unique jobs.",
            len(all_jobs),
            len(deduped),
        )

        # 5. Update cache with all deduped jobs if cache is present
        if self.cache is not None:
            self.cache.update(deduped)

        # 6. Apply preferences filtering & scoring if provided
        if preferences is not None:
            deduped = filter_jobs(deduped, preferences)

        return deduped

    async def check_all_health(self) -> dict[str, bool]:
        """Check operational health across all registered sources concurrently.

        Returns:
            dict[str, bool]: Mapping of source_id -> is_healthy.
        """
        all_sources = self.registry.get_all()
        if not all_sources:
            return {}

        tasks = [source.check_health() for source in all_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_map: dict[str, bool] = {}
        for source, res in zip(all_sources, results):
            if isinstance(res, Exception) or not isinstance(res, bool):
                logger.warning("Health check exception for source '%s': %s", source.source_id, res)
                health_map[source.source_id] = False
            else:
                health_map[source.source_id] = res

        return health_map
