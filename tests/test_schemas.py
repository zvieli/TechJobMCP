"""Unit tests for data models and schemas in job_mcp.models.schemas."""

import json
from job_mcp.models.schemas import Job, WorkMode


def test_job_default_structured_sections_and_semantic_score() -> None:
    """Default instantiation of Job should have structured section fields and semantic_score as None."""
    job = Job(
        job_id="job-123",
        title="Senior Python Engineer",
        company="TechCorp",
    )
    assert job.requirements is None
    assert job.responsibilities is None
    assert job.company_overview is None
    assert job.semantic_score is None
    assert job.description == ""


def test_job_explicit_structured_sections_and_semantic_score() -> None:
    """Explicit instantiation of Job should preserve section fields and semantic_score."""
    job = Job(
        job_id="job-456",
        title="Full Stack Developer",
        company="InnoTech",
        description="Full job description text",
        requirements="5+ years Python, FastAPI, React",
        responsibilities="Design APIs, build UI components",
        company_overview="Fast growing AI startup in Tel Aviv",
        semantic_score=88.5,
    )
    assert job.requirements == "5+ years Python, FastAPI, React"
    assert job.responsibilities == "Design APIs, build UI components"
    assert job.company_overview == "Fast growing AI startup in Tel Aviv"
    assert job.semantic_score == 88.5
    assert job.description == "Full job description text"


def test_job_serialization_roundtrip_with_sections() -> None:
    """Serialization to dict and JSON roundtrip preserves structured fields."""
    job = Job(
        job_id="job-789",
        title="ML Engineer",
        company="DataCo",
        work_mode=WorkMode.HYBRID,
        requirements="PyTorch, Transformers",
        responsibilities="Train LLMs",
        company_overview="Leading AI lab",
        semantic_score=94.2,
    )

    data = job.model_dump()
    assert data["requirements"] == "PyTorch, Transformers"
    assert data["responsibilities"] == "Train LLMs"
    assert data["company_overview"] == "Leading AI lab"
    assert data["semantic_score"] == 94.2

    # Reconstruct from dict
    restored_job = Job.model_validate(data)
    assert restored_job == job

    # JSON roundtrip
    json_str = job.model_dump_json()
    loaded_dict = json.loads(json_str)
    assert loaded_dict["requirements"] == "PyTorch, Transformers"
    assert loaded_dict["responsibilities"] == "Train LLMs"
    assert loaded_dict["company_overview"] == "Leading AI lab"
    assert loaded_dict["semantic_score"] == 94.2

    restored_from_json = Job.model_validate_json(json_str)
    assert restored_from_json == job
