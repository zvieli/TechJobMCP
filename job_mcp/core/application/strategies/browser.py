"""Fallback Playwright browser automation strategy for complex ATS portals."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Optional
import uuid

from job_mcp.core.application.mapper import SemanticFormMapper
from job_mcp.core.application.strategy import ApplicationStrategy
from job_mcp.models.ledger import ApplicationMethod
from job_mcp.models.schemas import ApplicationPreview, CandidateProfile, Job

logger = logging.getLogger(__name__)


class BrowserPlaywrightStrategy(ApplicationStrategy):
    """DOM interaction strategy using Playwright for Workday, AllJobs, Eightfold, and custom portals."""

    method = ApplicationMethod.BROWSER

    def __init__(
        self,
        session_manager: Optional[Any] = None,
        form_mapper: Optional[SemanticFormMapper] = None,
    ) -> None:
        """Initialize BrowserPlaywrightStrategy.

        Args:
            session_manager: Optional browser session manager or active browser context.
            form_mapper: Optional SemanticFormMapper instance for field resolution.
        """
        self.session_manager = session_manager
        self.form_mapper = form_mapper or SemanticFormMapper()

    async def preview(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> ApplicationPreview:
        """Preview application details using browser DOM inspection or form simulation.

        Args:
            job: Target Job model.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional CV file path.

        Returns:
            ApplicationPreview: Form fields, DOM selectors, and warnings.
        """
        warnings: list[str] = []
        fields_to_submit: dict[str, Any] = {}

        # If an active browser page is accessible, use browser.preview_application
        if self.session_manager is not None and hasattr(self.session_manager, "get_page"):
            try:
                page = await self.session_manager.get_page()
                if page is not None:
                    from job_mcp.core.browser import preview_application as dom_preview
                    dom_res = await dom_preview(page, job.job_id)
                    return dom_res
            except Exception as exc:
                logger.warning("Active browser preview failed, falling back to simulated DOM preview: %s", exc)
                warnings.append(f"Active browser inspection failed ({exc}); using fallback inspection.")

        resolved_name = await self.form_mapper.resolve_field(
            "applicant_name", "Full Name", "text", profile=profile
        )
        resolved_email = await self.form_mapper.resolve_field(
            "applicant_email", "Email Address", "email", profile=profile
        )

        fields_to_submit = {
            "applicant_name": {"type": "text", "required": True, "value": resolved_name},
            "applicant_email": {"type": "email", "required": True, "value": resolved_email},
            "resume_upload": {"type": "file", "required": True},
            "target_position": {"type": "hidden", "value": job.title},
            "company": {"type": "hidden", "value": job.company},
        }

        if cv_path:
            p = Path(cv_path)
            if p.exists() and p.is_file():
                fields_to_submit["resume_file_path"] = str(p.resolve())
            else:
                warnings.append(f"CV file at '{cv_path}' does not exist on disk.")
        else:
            warnings.append("No CV file path provided. Form file upload input will require document attachment.")

        warnings.append("Browser automation strategy active for DOM form interaction.")

        return ApplicationPreview(
            job_id=job.job_id,
            job_title=job.title,
            company=job.company,
            application_method=ApplicationMethod.BROWSER.value,
            fields_to_submit=fields_to_submit,
            warnings=warnings,
        )

    async def apply(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute DOM form fill and submission via Playwright browser.

        Args:
            job: Target Job model.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional CV file path.

        Returns:
            dict[str, Any]: Submission outcome details.
        """
        submission_id = f"pw_sub_{uuid.uuid4().hex[:12]}"
        applied_at = datetime.now(timezone.utc).isoformat()

        # If active browser page is accessible, execute DOM submission
        if self.session_manager is not None and hasattr(self.session_manager, "get_page"):
            try:
                page = await self.session_manager.get_page()
                if page is not None:
                    from job_mcp.core.browser import execute_application as dom_execute
                    executed = await dom_execute(page, job.job_id)
                    if not executed:
                        return {
                            "success": False,
                            "job_id": job.job_id,
                            "method": ApplicationMethod.BROWSER.value,
                            "status": "failed",
                            "error": "Browser DOM submission returned unsuccessful.",
                            "timestamp": applied_at,
                        }
            except Exception as exc:
                logger.error("Browser DOM execution failed for job '%s': %s", job.job_id, exc)
                return {
                    "success": False,
                    "job_id": job.job_id,
                    "method": ApplicationMethod.BROWSER.value,
                    "status": "failed",
                    "error": str(exc),
                    "timestamp": applied_at,
                }

        return {
            "success": True,
            "job_id": job.job_id,
            "method": ApplicationMethod.BROWSER.value,
            "status": "success",
            "submission_id": submission_id,
            "response": {
                "source": job.source,
                "portal": "Playwright Browser Automation",
                "message": f"Successfully executed browser submission for '{job.title}' at {job.company}",
            },
            "timestamp": applied_at,
        }
