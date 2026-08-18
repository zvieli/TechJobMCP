"""Multi-source job aggregator coordinating concurrent fetching, deduplication, and ranking."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from job_mcp.core.api_client import JobCache, filter_jobs
from job_mcp.models.schemas import Job, JobPreferences
from job_mcp.sources.base import BaseJobSource
from job_mcp.sources.dedup import deduplicate_jobs
from job_mcp.utils.logger import get_logger

if TYPE_CHECKING:
    from job_mcp.sources import SourceRegistry

logger = get_logger(__name__)


class JobAggregator:
    """Aggregates and deduplicates job listings across multiple registered sources."""

    def __init__(
        self,
        registry: Optional[SourceRegistry] = None,
        cache: Optional[JobCache] = None,
    ) -> None:
        """Initialize JobAggregator.

        Args:
            registry: Optional SourceRegistry instance. If None, uses default registry.
            cache: Optional JobCache instance for storing aggregated jobs.
        """
        if registry is None:
            from job_mcp.sources import registry as default_registry

            self.registry = default_registry
        else:
            self.registry = registry

        self.cache = cache

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

        # 3. Fetch from all sources concurrently with error isolation
        logger.info(
            "Fetching jobs concurrently from %d source(s): %s",
            len(active_sources),
            [s.source_id for s in active_sources],
        )

        tasks = [
            source.fetch_jobs(preferences=preferences, limit=limit_per_source)
            for source in active_sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_jobs: list[Job] = []
        for source, res in zip(active_sources, results):
            if isinstance(res, Exception):
                logger.warning(
                    "Source '%s' fetch_jobs failed: %s",
                    source.source_id,
                    res,
                    exc_info=res,
                )
            elif isinstance(res, list):
                logger.info("Source '%s' returned %d jobs.", source.source_id, len(res))
                all_jobs.extend(res)
            else:
                logger.warning(
                    "Source '%s' returned unexpected result type: %s",
                    source.source_id,
                    type(res),
                )

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
