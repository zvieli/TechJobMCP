"""Multi-source job aggregator coordinating concurrent fetching, deduplication, and ranking."""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any, Optional

from job_mcp.core.api_client import JobCache, extract_candidate_profile, filter_jobs
from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences
from job_mcp.sources.base import BaseJobSource
from job_mcp.sources.dedup import deduplicate_jobs
from job_mcp.utils.logger import get_logger

if TYPE_CHECKING:
    from job_mcp.sources import SourceRegistry

logger = get_logger(__name__)


DEFAULT_SOURCE_TIMEOUT: float = float(os.getenv("SOURCE_TIMEOUT_SECONDS", "6.0"))


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
        profile: Optional[CandidateProfile] = None,
    ) -> list[Job]:
        """Fetch job listings concurrently from active registered sources, deduplicate, and score.

        Args:
            sources: Optional list of source IDs to query (e.g. ['hiremetech', 'comeet', 'alljobs']).
                     If None, fetches from all active registered sources.
            preferences: Optional JobPreferences for filtering, keyword matching, and match scoring.
            limit_per_source: Maximum number of jobs to fetch per individual source.
            force_refresh: If True, bypasses cache and forces live fetching across sources.
            profile: Optional pre-extracted CandidateProfile for dynamic search query propagation and scoring.

        Returns:
            list[Job]: Unified, deduplicated, and optionally scored list of Job listings.
        """
        # Resolve CandidateProfile if not explicitly passed and preferences contains cv_path
        if profile is None and preferences is not None and preferences.cv_path:
            profile = extract_candidate_profile(preferences.cv_path)

        # Propagate dynamic queries / keywords / exclusions to child sources
        child_preferences = preferences
        if profile is not None:
            primary_stack = profile.primary_stack or profile.top_skills or profile.skills
            if preferences is None:
                preferences = JobPreferences(
                    tech_stack=list(primary_stack),
                    keywords=list(profile.search_queries[:3] or profile.top_skills[:3]),
                    exclude_keywords=list(profile.suggested_exclusions),
                )
                child_preferences = preferences
            else:
                effective_keywords = (
                    list(preferences.keywords)
                    if preferences.keywords
                    else list(profile.search_queries[:3] or profile.top_skills[:3])
                )
                effective_tech = (
                    list(preferences.tech_stack)
                    if preferences.tech_stack
                    else list(primary_stack)
                )
                effective_exclude = (
                    list(preferences.exclude_keywords)
                    if preferences.exclude_keywords
                    else list(profile.suggested_exclusions)
                )
                child_preferences = preferences.model_copy(
                    update={
                        "keywords": effective_keywords,
                        "tech_stack": effective_tech,
                        "exclude_keywords": effective_exclude,
                    }
                )
                preferences = child_preferences

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
                if preferences is not None or profile is not None:
                    results = filter_jobs(results, preferences or JobPreferences(), profile=profile)
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
                    src.fetch_jobs(preferences=child_preferences, limit=limit_per_source),
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
            *[_fetch_with_timeout(s) for s in active_sources],
            return_exceptions=True,
        )

        all_jobs: list[Job] = []
        for src, res in zip(active_sources, results):
            if isinstance(res, Exception):
                logger.warning("Source '%s' fetch failed: %s", src.source_id, res)
            elif isinstance(res, list):
                if res:
                    logger.info("Source '%s' returned %d jobs.", src.source_id, len(res))
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
        if preferences is not None or profile is not None:
            deduped = filter_jobs(deduped, preferences or JobPreferences(), profile=profile)

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
