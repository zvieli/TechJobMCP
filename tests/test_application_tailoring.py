"""Unit tests for System 2 Application Tailoring."""

import pytest
from job_mcp.core.application.tailoring import ApplicationPackage, generate_application_package
from job_mcp.models.schemas import CandidateProfile, Job, WorkMode


@pytest.mark.asyncio
async def test_generate_application_package_fallback():
    """Verify that generate_application_package produces a valid ApplicationPackage even without live LLM."""
    job = Job(
        job_id="test_tailor_1",
        title="Software Engineer",
        company="Comm-IT",
        location="Petah Tikva, IL",
        work_mode=WorkMode.ONSITE,
        tech_stack=["Python", "React", "Docker", "SQL"],
        description="Looking for a junior software engineer to develop platform services.",
    )

    profile = CandidateProfile(
        full_name="Lior Zvieli",
        seniority_level="Junior",
        top_skills=["Python", "React", "Docker", "AI", "LangGraph"],
        target_roles=["Software Engineer", "Full Stack Engineer"],
    )

    package = await generate_application_package(job, profile)

    assert isinstance(package, ApplicationPackage)
    assert package.job_id == "test_tailor_1"
    assert package.company == "Comm-IT"
    assert len(package.tailored_cv_highlights) >= 2
    assert "Comm-IT" in package.custom_cover_letter or "Software Engineer" in package.custom_cover_letter
    assert len(package.recruiter_pitch) > 10
    assert len(package.interview_prep_questions) >= 2
    for q in package.interview_prep_questions:
        assert q.question
        assert q.topic
        assert q.recommended_strategy
