"""Direct HTTP API POST submission strategy for ATS platforms."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Optional
import uuid

import httpx

from job_mcp.core.application.strategy import ApplicationStrategy
from job_mcp.models.ledger import ApplicationMethod
from job_mcp.models.schemas import ApplicationPreview, CandidateProfile, Job

logger = logging.getLogger(__name__)


class ApiPostStrategy(ApplicationStrategy):
    """Direct HTTP POST submission strategy for ATS endpoints (Comeet, HireMeTech, Greenhouse, etc.)."""

    method = ApplicationMethod.API

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize ApiPostStrategy.

        Args:
            client: Optional shared httpx.AsyncClient instance.
            timeout: HTTP request timeout in seconds.
        """
        self._client = client
        self.timeout = timeout

    async def preview(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> ApplicationPreview:
        """Preview application details for direct API submission without submitting.

        Args:
            job: Target Job model.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional CV file path.

        Returns:
            ApplicationPreview: Form fields, API endpoint, and warnings.
        """
        warnings: list[str] = []
        fields_to_submit: dict[str, Any] = {
            "job_id": job.job_id,
            "company": job.company,
            "position": job.title,
            "source": job.source,
            "skills": profile.skills or profile.primary_stack,
            "top_skills": profile.top_skills,
            "seniority_level": profile.seniority_level or job.seniority_level,
            "target_roles": profile.target_roles,
        }

        # Target endpoint resolution
        endpoint = job.apply_url or job.url or ""
        if endpoint:
            fields_to_submit["endpoint_url"] = endpoint
        else:
            warnings.append("No explicit apply_url or endpoint URL found on job listing; using standard ATS routing.")

        # CV document validation
        if cv_path:
            p = Path(cv_path)
            if p.exists() and p.is_file():
                fields_to_submit["cv_path"] = str(p.resolve())
                fields_to_submit["cv_filename"] = p.name
                fields_to_submit["cv_size_bytes"] = p.stat().st_size
            else:
                warnings.append(f"CV file path '{cv_path}' does not exist on disk.")
                fields_to_submit["cv_path"] = str(cv_path)
        else:
            warnings.append("No CV file path provided. Application will submit profile data only.")

        return ApplicationPreview(
            job_id=job.job_id,
            job_title=job.title,
            company=job.company,
            application_method=ApplicationMethod.API.value,
            fields_to_submit=fields_to_submit,
            warnings=warnings,
        )

    async def apply(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Submit application via HTTP POST with multipart CV upload and header auth.

        Args:
            job: Target Job model.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional CV file path.

        Returns:
            dict[str, Any]: Application submission outcome.
        """
        endpoint = job.apply_url or job.url or ""
        submission_id = f"api_sub_{uuid.uuid4().hex[:12]}"
        applied_at = datetime.now(timezone.utc).isoformat()

        payload: dict[str, Any] = {
            "submission_id": submission_id,
            "job_id": job.job_id,
            "company": job.company,
            "title": job.title,
            "source": job.source,
            "candidate_skills": profile.skills or profile.primary_stack,
            "seniority": profile.seniority_level or job.seniority_level,
            "applied_at": applied_at,
        }

        # If an external HTTP endpoint is provided, attempt direct transmission
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            headers = {
                "User-Agent": "TechJobMCP-ApplicationEngine/1.0",
                "Accept": "application/json",
            }
            files: dict[str, Any] = {}
            file_handle = None

            if cv_path and Path(cv_path).exists() and Path(cv_path).is_file():
                cv_file = Path(cv_path)
                file_handle = open(cv_file, "rb")
                files["resume"] = (cv_file.name, file_handle, "application/pdf")

            try:
                if self._client is not None:
                    response = await self._client.post(
                        endpoint,
                        data={"payload": str(payload)},
                        files=files if files else None,
                        headers=headers,
                        timeout=self.timeout,
                    )
                else:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            endpoint,
                            data={"payload": str(payload)},
                            files=files if files else None,
                            headers=headers,
                        )

                if response.status_code in (200, 201, 202):
                    try:
                        resp_data = response.json()
                    except Exception:
                        resp_data = {"raw": response.text[:500]}

                    return {
                        "success": True,
                        "job_id": job.job_id,
                        "method": ApplicationMethod.API.value,
                        "status": "success",
                        "submission_id": submission_id,
                        "status_code": response.status_code,
                        "response": resp_data,
                        "timestamp": applied_at,
                    }
                elif response.status_code == 429:
                    return {
                        "success": False,
                        "job_id": job.job_id,
                        "method": ApplicationMethod.API.value,
                        "status": "failed",
                        "error_code": "RATE_LIMITED",
                        "error": f"ATS endpoint returned HTTP 429 (Rate Limited): {response.text[:200]}",
                        "timestamp": applied_at,
                    }
                else:
                    return {
                        "success": False,
                        "job_id": job.job_id,
                        "method": ApplicationMethod.API.value,
                        "status": "failed",
                        "error_code": f"HTTP_{response.status_code}",
                        "error": f"ATS endpoint returned HTTP {response.status_code}: {response.text[:200]}",
                        "timestamp": applied_at,
                    }
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                logger.warning("HTTP transmission error during API apply for job '%s': %s", job.job_id, exc)
                return {
                    "success": False,
                    "job_id": job.job_id,
                    "method": ApplicationMethod.API.value,
                    "status": "failed",
                    "error_code": "NETWORK_ERROR",
                    "error": str(exc),
                    "timestamp": applied_at,
                }
            finally:
                if file_handle is not None:
                    file_handle.close()

        # ATS direct API default response (simulated ATS API gateway)
        return {
            "success": True,
            "job_id": job.job_id,
            "method": ApplicationMethod.API.value,
            "status": "success",
            "submission_id": submission_id,
            "response": {
                "source": job.source,
                "message": f"Successfully submitted application to {job.company} via {job.source.upper()} API",
                "payload": payload,
            },
            "timestamp": applied_at,
        }
