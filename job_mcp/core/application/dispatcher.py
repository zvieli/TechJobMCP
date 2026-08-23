"""Hybrid application dispatcher enforcing safety guardrails and routing strategies."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import re
from typing import Any, Optional

from job_mcp.core.application.ledger_service import ApplicationLedger
from job_mcp.core.application.strategy import ApplicationStrategy, get_application_strategy
from job_mcp.models.ledger import ApplicationEntry, ApplicationMethod, ApplicationStatus
from job_mcp.models.schemas import ApplicationPreview, CandidateProfile, Job, WorkMode

logger = logging.getLogger(__name__)

# Regex pattern matching Israel, Israeli cities/regions, IL country code, and Remote
ISRAEL_LOCATION_PATTERN = re.compile(
    r"\b(israel|il|remote|tel[\s\-_]*aviv|herzliya|haifa|jerusalem|rehovot|ramat[\s\-_]*gan|raanana|ra'anana|petah[\s\-_]*tikva|petach[\s\-_]*tikva|beer[\s\-_]*sheva|beersheba|yokneam|yokne'am|netanya|kfar[\s\-_]*saba|hod[\s\-_]*hasharon|holon|bat[\s\-_]*yam|modiin|modi'in|rishon[\s\-_]*lezion|givatayim|caesarea|ness[\s\-_]*ziona|bnei[\s\-_]*brak|glilot)\b",
    re.IGNORECASE,
)


class HybridApplicationDispatcher:
    """Central routing dispatcher for job application submissions with safety guardrails."""

    def __init__(
        self,
        ledger: Optional[ApplicationLedger] = None,
        session_manager: Optional[Any] = None,
        max_daily_applications: Optional[int] = None,
        min_match_score: float = 85.0,
    ) -> None:
        """Initialize the dispatcher with audit ledger and configurable guardrails.

        Args:
            ledger: Optional ApplicationLedger instance (initializes default if None).
            session_manager: Optional browser session manager or active browser context.
            max_daily_applications: Optional maximum allowed successful applications per day (defaults to MAX_DAILY_APPLICATIONS env or 10).
            min_match_score: Minimum match score threshold for autonomous apply (default 85.0).
        """
        self.ledger = ledger if ledger is not None else ApplicationLedger()
        self.session_manager = session_manager
        self.min_match_score = float(min_match_score)

        if max_daily_applications is not None:
            self.max_daily_applications = int(max_daily_applications)
        else:
            env_cap = os.getenv("MAX_DAILY_APPLICATIONS", "10").strip()
            try:
                self.max_daily_applications = int(env_cap)
            except ValueError:
                self.max_daily_applications = 10

    def _is_auto_apply_enabled(self) -> bool:
        """Check if autonomous application is enabled via fail-closed environment variable."""
        val = os.getenv("AUTO_APPLY_ENABLED", "false").strip().lower()
        return val in ("true", "1", "yes", "on")

    def _validate_location(self, job: Job) -> bool:
        """Validate that the job location matches Israel / IL / Remote constraints.

        Args:
            job: Target Job listing.

        Returns:
            bool: True if location is valid for autonomous application, False otherwise.
        """
        if job.work_mode == WorkMode.REMOTE:
            return True

        location_str = str(job.location or "").strip()
        if not location_str:
            # If no location specified and work mode not remote, fail constraint
            return False

        return bool(ISRAEL_LOCATION_PATTERN.search(location_str))

    def _validate_match_score(self, job: Job) -> bool:
        """Validate that the job match score meets the minimum threshold.

        Args:
            job: Target Job listing.

        Returns:
            bool: True if score >= min_match_score (supports 0-100 and 0-1 scales).
        """
        if job.match_score is None:
            return False

        score = float(job.match_score)
        if 0.0 < score <= 1.0:
            score = score * 100.0

        return score >= self.min_match_score

    async def preview_application(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> ApplicationPreview:
        """Preview application fields and check guardrail status without submitting.

        Args:
            job: Target Job listing.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional CV file path.

        Returns:
            ApplicationPreview: Form fields, submission strategy, and guardrail warnings.
        """
        strategy = get_application_strategy(job.source, session_manager=self.session_manager)
        preview = await strategy.preview(job, profile, cv_path=cv_path)

        # Inspect safety guardrails and append diagnostic warnings
        if self.ledger.is_applied(job.job_id):
            preview.warnings.insert(
                0,
                f"Duplicate Guardrail: Job '{job.job_id}' has already been successfully applied to in ledger.",
            )

        if not self._is_auto_apply_enabled():
            preview.warnings.append(
                "Guardrail Alert: Autonomous application is disabled (AUTO_APPLY_ENABLED=false). Explicit force=True required to submit."
            )

        daily_count = self.ledger.get_daily_count()
        if daily_count >= self.max_daily_applications:
            preview.warnings.append(
                f"Guardrail Alert: Daily application cap reached ({daily_count}/{self.max_daily_applications})."
            )

        if not self._validate_match_score(job):
            preview.warnings.append(
                f"Guardrail Alert: Match score ({job.match_score}) is below required threshold ({self.min_match_score})."
            )

        if not self._validate_location(job):
            preview.warnings.append(
                f"Guardrail Alert: Location '{job.location}' does not match Israel/IL/Remote constraint."
            )

        return preview

    async def execute_application(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Execute job application submission enforcing all safety guardrails.

        Guardrail order:
        1. Duplicate submission check (via ledger.is_applied).
        2. Fail-closed AUTO_APPLY_ENABLED check (unless force=True).
        3. Daily application cap check (unless force=True).
        4. Match score threshold check (>= 85.0 unless force=True).
        5. Location constraint check (Israel / IL / Remote unless force=True).

        Args:
            job: Target Job listing.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional path to CV document.
            force: If True, bypasses non-duplicate guardrails (auto apply toggle, daily cap, score, location).

        Returns:
            dict[str, Any]: Application submission outcome.
        """
        # Resolve strategy early to tag ledger method properly
        strategy = get_application_strategy(job.source, session_manager=self.session_manager)
        method = strategy.method

        # 1. Duplicate check (Always enforced, even with force=True)
        if self.ledger.is_applied(job.job_id):
            msg = f"Duplicate application blocked: Job '{job.job_id}' has already been applied to."
            logger.warning(msg)
            self.ledger.record_application(
                ApplicationEntry(
                    job_id=job.job_id,
                    company=job.company,
                    job_title=job.title,
                    source=job.source,
                    method=method,
                    status=ApplicationStatus.BLOCKED,
                    match_score=job.match_score,
                    cv_used=cv_path,
                    error_message=msg,
                    notes="Blocked by duplicate submission guardrail",
                )
            )
            return {
                "success": False,
                "job_id": job.job_id,
                "status": ApplicationStatus.BLOCKED.value,
                "error_code": "DUPLICATE_APPLICATION",
                "message": msg,
            }

        # 2. AUTO_APPLY_ENABLED check
        if not self._is_auto_apply_enabled() and not force:
            msg = "Autonomous application is disabled (AUTO_APPLY_ENABLED=false). Use force=True to override."
            logger.warning("Guardrail blocked job '%s': %s", job.job_id, msg)
            self.ledger.record_application(
                ApplicationEntry(
                    job_id=job.job_id,
                    company=job.company,
                    job_title=job.title,
                    source=job.source,
                    method=method,
                    status=ApplicationStatus.BLOCKED,
                    match_score=job.match_score,
                    cv_used=cv_path,
                    error_message=msg,
                    notes="Blocked by fail-closed AUTO_APPLY_ENABLED guardrail",
                )
            )
            return {
                "success": False,
                "job_id": job.job_id,
                "status": ApplicationStatus.BLOCKED.value,
                "error_code": "AUTO_APPLY_DISABLED",
                "message": msg,
            }

        # 3. Daily application cap check
        daily_count = self.ledger.get_daily_count()
        if daily_count >= self.max_daily_applications and not force:
            msg = f"Daily application cap reached ({daily_count}/{self.max_daily_applications}). Submission blocked."
            logger.warning("Guardrail blocked job '%s': %s", job.job_id, msg)
            self.ledger.record_application(
                ApplicationEntry(
                    job_id=job.job_id,
                    company=job.company,
                    job_title=job.title,
                    source=job.source,
                    method=method,
                    status=ApplicationStatus.BLOCKED,
                    match_score=job.match_score,
                    cv_used=cv_path,
                    error_message=msg,
                    notes="Blocked by daily application cap guardrail",
                )
            )
            return {
                "success": False,
                "job_id": job.job_id,
                "status": ApplicationStatus.BLOCKED.value,
                "error_code": "DAILY_CAP_REACHED",
                "message": msg,
            }

        # 4. Match score threshold check
        if not self._validate_match_score(job) and not force:
            msg = f"Job match score ({job.match_score}) is below the required threshold of {self.min_match_score}."
            logger.warning("Guardrail blocked job '%s': %s", job.job_id, msg)
            self.ledger.record_application(
                ApplicationEntry(
                    job_id=job.job_id,
                    company=job.company,
                    job_title=job.title,
                    source=job.source,
                    method=method,
                    status=ApplicationStatus.BLOCKED,
                    match_score=job.match_score,
                    cv_used=cv_path,
                    error_message=msg,
                    notes="Blocked by match score threshold guardrail",
                )
            )
            return {
                "success": False,
                "job_id": job.job_id,
                "status": ApplicationStatus.BLOCKED.value,
                "error_code": "LOW_MATCH_SCORE",
                "message": msg,
            }

        # 5. Location constraint check
        if not self._validate_location(job) and not force:
            msg = f"Job location '{job.location}' does not match Israel/IL/Remote constraints."
            logger.warning("Guardrail blocked job '%s': %s", job.job_id, msg)
            self.ledger.record_application(
                ApplicationEntry(
                    job_id=job.job_id,
                    company=job.company,
                    job_title=job.title,
                    source=job.source,
                    method=method,
                    status=ApplicationStatus.BLOCKED,
                    match_score=job.match_score,
                    cv_used=cv_path,
                    error_message=msg,
                    notes="Blocked by location constraint guardrail",
                )
            )
            return {
                "success": False,
                "job_id": job.job_id,
                "status": ApplicationStatus.BLOCKED.value,
                "error_code": "LOCATION_CONSTRAINT_FAILED",
                "message": msg,
            }

        # All guardrails passed -> Delegate to strategy
        logger.info(
            "All guardrails passed for job '%s' (source=%s). Delegating to %s.",
            job.job_id,
            job.source,
            strategy.__class__.__name__,
        )

        try:
            result = await strategy.apply(job, profile, cv_path=cv_path)
            is_success = bool(result.get("success", False))
            status = ApplicationStatus.SUCCESS if is_success else ApplicationStatus.FAILED

            self.ledger.record_application(
                ApplicationEntry(
                    job_id=job.job_id,
                    company=job.company,
                    job_title=job.title,
                    source=job.source,
                    method=method,
                    status=status,
                    match_score=job.match_score,
                    cv_used=cv_path,
                    response_payload=result.get("response") or result,
                    error_message=result.get("error") if not is_success else None,
                    notes="Autonomous application dispatch",
                )
            )
            return result

        except Exception as exc:
            logger.exception("Unexpected error executing application strategy for job '%s': %s", job.job_id, exc)
            self.ledger.record_application(
                ApplicationEntry(
                    job_id=job.job_id,
                    company=job.company,
                    job_title=job.title,
                    source=job.source,
                    method=method,
                    status=ApplicationStatus.FAILED,
                    match_score=job.match_score,
                    cv_used=cv_path,
                    error_message=str(exc),
                    notes="Strategy execution exception",
                )
            )
            return {
                "success": False,
                "job_id": job.job_id,
                "status": ApplicationStatus.FAILED.value,
                "error_code": "EXECUTION_ERROR",
                "message": str(exc),
            }
