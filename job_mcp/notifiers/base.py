"""Base abstract class for job alert notifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from job_mcp.models.schemas import Job


class BaseNotifier(ABC):
    """Abstract base class for all job notification channels."""

    @abstractmethod
    def format_alert(self, jobs: list[Job], title: Optional[str] = None) -> str:
        """Format a list of jobs into a notification message string."""
        pass

    @abstractmethod
    async def send_alert(self, jobs: list[Job], title: Optional[str] = None) -> bool:
        """Send notification alert for a batch of jobs.

        Args:
            jobs: List of Job models to notify about.
            title: Optional title/header for the alert.

        Returns:
            bool: True if alert was sent successfully (or empty jobs), False otherwise.
        """
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """Check connection health and authentication status of the notifier.

        Returns:
            bool: True if channel is operational, False otherwise.
        """
        pass
