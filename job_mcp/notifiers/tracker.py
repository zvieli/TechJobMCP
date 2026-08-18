"""Job tracker for deduplicating alerts and tracking seen job postings."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from job_mcp.models.schemas import Job

logger = logging.getLogger(__name__)


class JobTracker:
    """Tracks seen job postings to prevent duplicate alert notifications."""

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
    ) -> None:
        """Initialize JobTracker.

        Args:
            storage_path: Optional path to JSON persistence file.
            auto_save: Whether to automatically persist changes to storage_path on mutation.
        """
        self.storage_path: Optional[Path] = Path(storage_path) if storage_path else None
        self.auto_save: bool = auto_save
        self._seen: set[str] = set()

        if self.storage_path and self.storage_path.exists():
            self.load()

    def _get_key(self, job_or_id: Union[Job, str]) -> str:
        """Extract a unique string key from a Job or job ID."""
        if isinstance(job_or_id, Job):
            if job_or_id.job_id:
                return str(job_or_id.job_id)
            return f"{job_or_id.company}:{job_or_id.title}"
        return str(job_or_id)

    def is_seen(self, job_or_id: Union[Job, str]) -> bool:
        """Check if a job has already been seen."""
        key = self._get_key(job_or_id)
        return key in self._seen

    def mark_seen(self, job_or_id: Union[Job, str]) -> bool:
        """Mark a job as seen.

        Returns:
            bool: True if newly marked as seen, False if already seen.
        """
        key = self._get_key(job_or_id)
        if key in self._seen:
            return False

        self._seen.add(key)
        if self.auto_save and self.storage_path:
            self.save()
        return True

    def mark_many_seen(self, jobs_or_ids: list[Union[Job, str]]) -> int:
        """Mark multiple jobs as seen.

        Returns:
            int: Number of newly marked jobs.
        """
        new_count = 0
        for item in jobs_or_ids:
            key = self._get_key(item)
            if key not in self._seen:
                self._seen.add(key)
                new_count += 1

        if new_count > 0 and self.auto_save and self.storage_path:
            self.save()
        return new_count

    def filter_unseen(self, jobs: list[Job], auto_mark: bool = False) -> list[Job]:
        """Filter a list of jobs, returning only those not yet seen.

        Args:
            jobs: List of Job models.
            auto_mark: If True, marks returned unseen jobs as seen.

        Returns:
            list[Job]: List of previously unseen jobs.
        """
        unseen_jobs = [job for job in jobs if not self.is_seen(job)]

        if auto_mark and unseen_jobs:
            self.mark_many_seen(unseen_jobs)

        return unseen_jobs

    def save(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        """Persist seen job IDs to a JSON file."""
        target_path = Path(storage_path) if storage_path else self.storage_path
        if not target_path:
            return

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "seen_ids": sorted(list(self._seen)),
                "total_seen": len(self._seen),
                "last_saved_at": datetime.now(timezone.utc).isoformat(),
            }
            tmp_path = target_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(target_path)
            logger.debug(f"Saved {len(self._seen)} seen jobs to {target_path}")
        except Exception as exc:
            logger.error(f"Failed to save job tracker state to {target_path}: {exc}")

    def load(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        """Load seen job IDs from a JSON file."""
        target_path = Path(storage_path) if storage_path else self.storage_path
        if not target_path or not target_path.exists():
            return

        try:
            content = target_path.read_text(encoding="utf-8").strip()
            if not content:
                return

            data = json.loads(content)
            if isinstance(data, list):
                self._seen.update(str(x) for x in data)
            elif isinstance(data, dict):
                ids = data.get("seen_ids", [])
                if isinstance(ids, list):
                    self._seen.update(str(x) for x in ids)
            logger.debug(f"Loaded {len(self._seen)} seen jobs from {target_path}")
        except Exception as exc:
            logger.warning(f"Failed to load job tracker state from {target_path}: {exc}")

    def clear(self) -> None:
        """Clear all tracked seen jobs."""
        self._seen.clear()
        if self.auto_save and self.storage_path:
            self.save()

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics and status."""
        return {
            "total_seen": len(self._seen),
            "storage_path": str(self.storage_path) if self.storage_path else None,
            "auto_save": self.auto_save,
        }

    def __len__(self) -> int:
        return len(self._seen)

    def __contains__(self, job_or_id: Union[Job, str]) -> bool:
        return self.is_seen(job_or_id)

    def __iter__(self):
        return iter(self._seen)
