"""Search Plane Aggregator Adapter for Milestone 6 (M6-A2).

Connects existing SourceRegistry and JobAggregator to the provider-agnostic
Search Plane contracts (JobSearchRequest, JobSearchResultItem, JobRef, FetchResult).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from job_mcp.core.search_plane.models import (
    FetchResult,
    FetchStatus,
    JobRef,
    JobSearchRequest,
    JobSearchResultItem,
    JobSearchResultSet,
    SourceCapabilities,
)
from job_mcp.models.schemas import Job, JobPreferences

if TYPE_CHECKING:
    from job_mcp.core.api_client import JobCache
    from job_mcp.sources.aggregator import JobAggregator
    from job_mcp.sources.registry import SourceRegistry

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Capability Matrix for Current Sources
# ===========================================================================

SOURCE_CAPABILITY_MAP: dict[str, SourceCapabilities] = {
    "linkedin": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=True,
        supports_url_fetch=True,
        supports_query=True,
        supports_company_filter=False,
        supports_work_mode=True,
        supports_pagination=True,
    ),
    "greenhouse": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=False,
        supports_company_filter=True,
        supports_work_mode=False,
        supports_pagination=False,
    ),
    "lever": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=False,
        supports_company_filter=True,
        supports_work_mode=False,
        supports_pagination=False,
    ),
    "workday": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=True,
        supports_company_filter=True,
        supports_work_mode=False,
        supports_pagination=True,
    ),
    "eightfold": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=True,
        supports_company_filter=True,
        supports_work_mode=False,
        supports_pagination=True,
    ),
    "direct_tech": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=True,
        supports_company_filter=True,
        supports_work_mode=False,
        supports_pagination=True,
    ),
    "comeet": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=False,
        supports_company_filter=True,
        supports_work_mode=False,
        supports_pagination=False,
    ),
    "alljobs": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=False,
        supports_company_filter=False,
        supports_work_mode=False,
        supports_pagination=True,
    ),
    "gotfriends": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=False,
        supports_company_filter=False,
        supports_work_mode=False,
        supports_pagination=False,
    ),
    "jobify": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=False,
        supports_company_filter=False,
        supports_work_mode=False,
        supports_pagination=False,
    ),
    "hiremetech": SourceCapabilities(
        supports_search=True,
        supports_native_fetch=False,
        supports_url_fetch=False,
        supports_query=False,
        supports_company_filter=False,
        supports_work_mode=False,
        supports_pagination=False,
    ),
}


def get_source_capabilities(source_id: str) -> SourceCapabilities:
    """Return declared capabilities for a source family, failing closed for unknown sources."""
    norm = source_id.strip().lower()
    return SOURCE_CAPABILITY_MAP.get(norm, SourceCapabilities(supports_search=False))


# ===========================================================================
# 2. SearchPlaneAdapter Implementation
# ===========================================================================


class SearchPlaneAdapter:
    """Adapts existing JobAggregator and SourceRegistry into the Unified Search Plane."""

    def __init__(
        self,
        aggregator: JobAggregator | None = None,
        registry: SourceRegistry | None = None,
        cache: JobCache | None = None,
    ) -> None:
        """Initialize SearchPlaneAdapter with optional aggregator, registry, and cache."""
        if aggregator is not None:
            self.aggregator = aggregator
            self.registry: SourceRegistry | None = registry or getattr(aggregator, "registry", None)
            self.cache: JobCache | None = cache or getattr(aggregator, "cache", None)
        else:
            from job_mcp.sources.aggregator import JobAggregator as DefaultAggregator
            from job_mcp.sources.registry import registry as default_registry

            self.registry = registry or default_registry
            self.cache = cache
            self.aggregator = DefaultAggregator(registry=self.registry, cache=self.cache)

    def create_job_ref(self, job: Job) -> JobRef:
        """Derive a deterministic, opaque JobRef from a canonical Job instance."""
        source_family = (job.source or "unknown").strip().lower()
        job_id = str(job.job_id)

        account: str | None = None
        locator: str = job_id

        if source_family == "greenhouse":
            account = job.company.strip() if job.company else None
            locator = job_id.removeprefix("greenhouse_")
        elif source_family == "lever":
            account = job.company.strip() if job.company else None
            locator = job_id.removeprefix("lever_")
        elif source_family == "workday":
            parts = job_id.split("_")
            if len(parts) >= 3 and parts[0] == "workday":
                account = parts[1]
                locator = "_".join(parts[2:])
            else:
                account = job.company.strip() if job.company else None
                locator = job_id.removeprefix("workday_")
        elif source_family == "eightfold":
            parts = job_id.split("_")
            if len(parts) >= 3 and parts[0] == "eightfold":
                account = parts[1]
                locator = "_".join(parts[2:])
            else:
                account = job.company.strip() if job.company else None
                locator = job_id.removeprefix("eightfold_")
        elif source_family == "direct_tech":
            parts = job_id.split("_")
            if len(parts) >= 3 and parts[0] == "direct":
                account = parts[1]
                locator = "_".join(parts[2:])
            else:
                account = job.company.strip() if job.company else None
                locator = job_id
        elif source_family == "linkedin":
            account = None
            locator = job_id.removeprefix("linkedin_")
        else:
            account = None
            locator = job_id

        return JobRef(
            version=1,
            source_family=source_family,
            account=account,
            locator=locator,
        )

    async def search(self, request: JobSearchRequest) -> JobSearchResultSet:
        """Execute a provider-agnostic search request via JobAggregator."""
        keywords = [request.query] if request.query else []
        prefs = JobPreferences(
            tech_stack=list(request.tech_stack),
            work_mode=request.work_mode,
            location=request.location,
            keywords=keywords,
        )

        jobs = await self.aggregator.fetch_all_jobs(
            sources=request.sources,
            preferences=prefs,
            per_source_limit=request.limit,
        )

        # Apply company filter post-retrieval if specified
        if request.company:
            c_filter = request.company.strip().lower()
            jobs = [j for j in jobs if c_filter in (j.company or "").lower()]

        # Respect limit constraint
        if request.limit and len(jobs) > request.limit:
            jobs = jobs[: request.limit]

        items: list[JobSearchResultItem] = []
        for job in jobs:
            ref = self.create_job_ref(job)
            items.append(
                JobSearchResultItem(
                    ref=ref.encode(),
                    title=job.title,
                    company=job.company,
                    location=job.location or "",
                    work_mode=job.work_mode,
                    canonical_url=job.url or job.apply_url,
                    source_family=ref.source_family,
                    posted_date=job.posted_date,
                    retrieval_method="structured",
                )
            )

        return JobSearchResultSet(
            items=items,
            total_estimated=len(items),
        )

    async def fetch(self, ref: str | JobRef) -> FetchResult:
        """Fetch a single posting by its JobRef locator using factual retrieval resolution."""
        # 1. Parse / Validate JobRef
        job_ref: JobRef
        if isinstance(ref, str):
            try:
                job_ref = JobRef.decode(ref)
            except (ValueError, TypeError) as exc:
                return FetchResult(
                    status=FetchStatus.INVALID_REF,
                    ref=ref,
                    diagnostic=f"Failed to decode JobRef: {exc}",
                )
        elif isinstance(ref, JobRef):
            job_ref = ref
        else:
            return FetchResult(
                status=FetchStatus.INVALID_REF,
                diagnostic="ref must be a string or JobRef instance",
            )

        ref_str = str(job_ref)

        # 2. Check JobCache fast path
        if self.cache is not None:
            candidate_ids = [job_ref.locator, f"{job_ref.source_family}_{job_ref.locator}"]
            if job_ref.account:
                candidate_ids.append(f"{job_ref.source_family}_{job_ref.account}_{job_ref.locator}")
                candidate_ids.append(f"direct_{job_ref.account}_{job_ref.locator}")

            for cid in candidate_ids:
                cached_job = self.cache.get_by_id(cid)
                if cached_job is not None:
                    return FetchResult(
                        status=FetchStatus.FOUND,
                        job=cached_job,
                        ref=ref_str,
                    )

        # 3. Check native provider refetch if supported
        caps = get_source_capabilities(job_ref.source_family)
        if caps.supports_native_fetch and job_ref.source_family == "linkedin":
            linkedin_src: Any = self.registry.get("linkedin") if self.registry else None
            if linkedin_src and hasattr(linkedin_src, "fetch_job_details"):
                try:
                    details = await linkedin_src.fetch_job_details(job_ref.locator)
                    if details:
                        raw_id = job_ref.locator.replace("linkedin_", "")
                        hydrated_job = Job(
                            job_id=f"linkedin_{raw_id}",
                            source="linkedin",
                            sources=["linkedin"],
                            url=f"https://www.linkedin.com/jobs/view/{raw_id}",
                            title=details["title"],
                            company=details["company"],
                            location=details.get("location") or "",
                            description=details.get("description") or "",
                            work_mode=details.get("work_mode"),
                            tech_stack=details.get("tech_stack") or [],
                            posted_date=details.get("posted_date"),
                            apply_url=details.get("apply_url"),
                            seniority_level=details.get("seniority_level"),
                            department=details.get("department"),
                            requirements=details.get("requirements"),
                            responsibilities=details.get("responsibilities"),
                            company_overview=details.get("company_overview"),
                        )
                        if self.cache is not None:
                            self.cache.update([hydrated_job])
                        return FetchResult(
                            status=FetchStatus.FOUND,
                            job=hydrated_job,
                            ref=ref_str,
                        )
                    return FetchResult(
                        status=FetchStatus.NOT_FOUND,
                        ref=ref_str,
                        diagnostic=f"Posting {job_ref.locator!r} was not found on LinkedIn.",
                    )
                except Exception as exc:  # noqa: BLE001
                    return FetchResult(
                        status=FetchStatus.UPSTREAM_ERROR,
                        ref=ref_str,
                        diagnostic=f"Upstream error while fetching LinkedIn posting: {exc}",
                    )

        # 4. Fallback for sources without native refetch support
        return FetchResult(
            status=FetchStatus.UNSUPPORTED_REFETCH,
            ref=ref_str,
            diagnostic=f"Source family '{job_ref.source_family}' does not support native single-posting refetch.",
        )


__all__ = [
    "SOURCE_CAPABILITY_MAP",
    "SearchPlaneAdapter",
    "get_source_capabilities",
]
