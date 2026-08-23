"""Data models and enums for job application ledger and safety guardrails."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ApplicationStatus(str, Enum):
    """Status of an application submission."""
    SUCCESS = "success"
    FAILED = "failed"
    STAGED = "staged"
    BLOCKED = "blocked"


class ApplicationMethod(str, Enum):
    """Method used to submit the job application."""
    API = "api"
    EASY_APPLY = "easy_apply"
    BROWSER = "browser"


class ApplicationEntry(BaseModel):
    """Model representing an audit log entry for a job application."""
    job_id: str
    company: str
    job_title: str
    source: str = ""
    applied_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    method: ApplicationMethod | str
    status: ApplicationStatus | str
    match_score: Optional[float] = None
    cv_used: Optional[str] = None
    response_payload: Optional[dict[str, Any] | str] = None
    error_message: Optional[str] = None
    notes: Optional[str] = None
