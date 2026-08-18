"""Sources module for multi-source job fetching and deduplication."""

from hireme_mcp.sources.dedup import (
    normalize_title,
    normalize_company,
    compute_dedup_key,
    merge_job_entities,
    deduplicate_jobs,
)

__all__ = [
    "normalize_title",
    "normalize_company",
    "compute_dedup_key",
    "merge_job_entities",
    "deduplicate_jobs",
]
