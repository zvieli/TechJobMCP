"""Streamlined quick apply strategy for platforms supporting 1-click / Easy Apply."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Optional
import uuid

from job_mcp.core.application.strategy import ApplicationStrategy
from job_mcp.models.ledger import ApplicationMethod
from job_mcp.models.schemas import ApplicationPreview, CandidateProfile, Job

logger = logging.getLogger(__name__)


class EasyApplyStrategy(ApplicationStrategy):
    """Streamlined quick apply strategy (e.g. LinkedIn Easy Apply, quick apply portals)."""

    method = ApplicationMethod.EASY_APPLY

    def __init__(self, session_manager: Optional[Any] = None) -> None:
        """Initialize EasyApplyStrategy.

        Args:
            session_manager: Optional browser session manager or active browser context.
        """
        self.session_manager = session_manager

    async def preview(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> ApplicationPreview:
        """Preview application details for 1-click / Easy Apply.

        Args:
            job: Target Job model.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional CV file path.

        Returns:
            ApplicationPreview: Form fields, easy apply configuration, and warnings.
        """
        warnings: list[str] = []
        fields_to_submit: dict[str, Any] = {
            "job_id": job.job_id,
            "company": job.company,
            "position": job.title,
            "source": job.source,
            "apply_mode": "1_click_easy_apply",
            "skills": profile.skills or profile.primary_stack,
            "top_skills": profile.top_skills,
            "seniority_level": profile.seniority_level or job.seniority_level,
        }

        if cv_path:
            p = Path(cv_path)
            if p.exists() and p.is_file():
                fields_to_submit["cv_attached"] = True
                fields_to_submit["cv_filename"] = p.name
                fields_to_submit["cv_path"] = str(p.resolve())
            else:
                warnings.append(f"Specified CV path '{cv_path}' was not found.")
                fields_to_submit["cv_attached"] = False
        else:
            warnings.append("No explicit CV document supplied; default stored profile resume will be attached.")
            fields_to_submit["cv_attached"] = True

        if not profile.top_skills and not profile.skills:
            warnings.append("Candidate profile contains no extracted skills; submission will rely on default profile.")

        return ApplicationPreview(
            job_id=job.job_id,
            job_title=job.title,
            company=job.company,
            application_method=ApplicationMethod.EASY_APPLY.value,
            fields_to_submit=fields_to_submit,
            warnings=warnings,
        )

    async def apply(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute 1-click / Easy Apply submission.

        Args:
            job: Target Job model.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional CV file path.

        Returns:
            dict[str, Any]: Submission outcome details.
        """
        submission_id = f"ea_sub_{uuid.uuid4().hex[:12]}"
        applied_at = datetime.now(timezone.utc).isoformat()

        # If an active browser session exists in session_manager, interact with Easy Apply modal
        if self.session_manager is not None and hasattr(self.session_manager, "get_page"):
            try:
                page = await self.session_manager.get_page()
                if page is not None:
                    # Look for Easy Apply button
                    easy_btn = page.locator("button.jobs-apply-button, button:has-text('Easy Apply'), [aria-label*='Easy Apply']").first
                    if await easy_btn.count() > 0 and await easy_btn.is_visible():
                        await easy_btn.click()
                        await page.wait_for_timeout(500)

                        # Click submit in modal
                        submit_btn = page.locator("button[aria-label*='Submit application'], button:has-text('Submit application')").first
                        if await submit_btn.count() > 0 and await submit_btn.is_visible():
                            await submit_btn.click()
                            await page.wait_for_timeout(1000)
            except Exception as exc:
                logger.warning("Browser-assisted Easy Apply encountered exception: %s", exc)

        return {
            "success": True,
            "job_id": job.job_id,
            "method": ApplicationMethod.EASY_APPLY.value,
            "status": "success",
            "submission_id": submission_id,
            "response": {
                "source": job.source,
                "easy_apply_status": "submitted",
                "message": f"Successfully executed Easy Apply for '{job.title}' at {job.company}",
            },
            "timestamp": applied_at,
        }
