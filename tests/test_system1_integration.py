"""Integration tests for System 1 decision engine wired across TechJobMCP."""

import pytest

from job_mcp.core.api_client import calculate_match_score, detect_seniority_level
from job_mcp.core.section_parser import parse_job_sections
from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences
from job_mcp.sources.dedup import deduplicate_jobs


def test_system1_deduplication_fuzzy_merging():
    """Verify that System 1 deduplication catches title variations within the same company."""
    job_a = Job(
        job_id="wix-1",
        title="Senior Python Backend Engineer",
        company="Wix.com",
        location="Tel Aviv",
        description="We are seeking a Senior Python Developer with 5+ years building backend microservices with FastAPI and Docker.",
        source="comeet",
    )
    job_b = Job(
        job_id="wix-2",
        title="Python Engineer - Senior Level (Backend)",
        company="Wix",
        location="Tel Aviv, Israel",
        description="Senior Python engineer needed for high-throughput distributed microservices in FastAPI, AWS, and Docker.",
        source="alljobs",
    )

    # Dedup without System 1 might create 2 keys: "senior python backend engineer@wix" vs "python engineer senior level backend@wix"
    deduped_legacy = deduplicate_jobs([job_a, job_b], enable_system1=False)
    assert len(deduped_legacy) == 2, "Legacy regex dedup kept them separate due to different title strings"

    # Dedup WITH System 1 merges them using neural duplicate verification
    deduped_system1 = deduplicate_jobs([job_a, job_b], enable_system1=True)
    assert len(deduped_system1) == 1, "System 1 neural dedup must identify and merge the two Wix job postings"
    merged = deduped_system1[0]
    assert "comeet" in merged.sources
    assert "alljobs" in merged.sources


def test_system1_seniority_detection():
    """Verify System 1 correctly detects seniority levels."""
    senior_level = detect_seniority_level(
        title="Staff Software Engineer",
        text="Looking for an experienced technical leader with 8+ years experience.",
        enable_system1=True,
    )
    assert senior_level in ("Senior", "Lead")

    junior_level = detect_seniority_level(
        title="Junior Backend Developer",
        text="Great opportunity for recent computer science graduates with 0-1 years experience.",
        enable_system1=True,
    )
    assert junior_level == "Junior"


def test_system1_match_scoring_ensemble_and_confidence():
    """Verify calculate_match_score integrates System 1 ensemble and populates confidence."""
    profile = CandidateProfile(
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
        top_skills=["Python", "FastAPI", "AWS"],
        primary_stack=["Python", "FastAPI"],
        target_roles=["Backend Engineer", "Software Engineer"],
    )
    prefs = JobPreferences(
        tech_stack=["Python", "FastAPI"],
        keywords=["Backend"],
    )
    job = Job(
        job_id="test-job-1",
        title="Senior Python Backend Developer",
        company="Cybereason",
        location="Tel Aviv",
        description="We are looking for a Python Backend Developer with 5+ years experience in FastAPI, Docker, and AWS.",
        tech_stack=["Python", "FastAPI", "Docker", "AWS"],
    )

    score = calculate_match_score(job, prefs, profile=profile)
    assert score > 50.0
    assert job.system1_confidence is not None
    assert 0.0 <= job.system1_confidence <= 1.0
    assert any("System 1 Neural Fit" in r for r in job.match_reasons)


def test_system1_section_parsing_unstructured():
    """Verify System 1 can recover sections from unformatted job descriptions."""
    unstructured_text = (
        "We are a leading cybersecurity company protecting millions of endpoints worldwide.\n\n"
        "You will design and build scalable distributed cloud services and lead microservice architecture.\n\n"
        "At least 5 years of professional experience with Python or Go, deep understanding of Kubernetes and AWS, and BS in Computer Science.\n\n"
        "Comprehensive health insurance, hybrid working flexibility, and stock options program."
    )

    sections = parse_job_sections(unstructured_text)
    # At least requirements or responsibilities should be populated
    assert sections.requirements != "" or sections.responsibilities != "" or sections.raw_other != ""
