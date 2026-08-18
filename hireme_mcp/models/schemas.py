"""Data models and schemas for HireMeTech MCP server."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class WorkMode(str, Enum):
    """Work mode classification."""
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"


class OperationMode(str, Enum):
    """Server operation mode controlling autonomy level."""
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"



class Job(BaseModel):
    """Job listing model."""
    job_id: str
    title: str
    company: str
    location: str = ""
    work_mode: Optional[WorkMode] = None
    tech_stack: list[str] = Field(default_factory=list)
    description: str = ""
    salary_range: Optional[str] = None
    posted_date: Optional[str] = None
    url: Optional[str] = None
    is_bookmarked: bool = False
    match_score: Optional[float] = None
    source: str = "hiremetech"
    sources: list[str] = Field(default_factory=lambda: ["hiremetech"])
    apply_url: Optional[str] = None
    department: Optional[str] = None

    @model_validator(mode="after")
    def _sync_sources(self) -> "Job":
        if self.source and self.source != "hiremetech" and self.sources == ["hiremetech"]:
            self.sources = [self.source]
        elif self.source and self.source not in self.sources:
            self.sources.insert(0, self.source)
        return self


class JobPreferences(BaseModel):
    """Job search and matching preferences."""
    tech_stack: list[str] = Field(default_factory=list)
    work_mode: Optional[WorkMode] = None
    location: Optional[str] = None
    min_salary: Optional[int] = None
    keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    cv_path: Optional[str] = None


class ApplicationPreview(BaseModel):
    """Preview of job application details before submission."""
    job_id: str
    job_title: str
    company: str
    application_method: str
    fields_to_submit: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ToolResponse(BaseModel):
    """Standard response model for MCP tools."""
    success: bool
    message: str
    data: Optional[dict[str, Any] | list[Any] | Any] = None
    error_code: Optional[str] = None
    trace_id: Optional[str] = None
