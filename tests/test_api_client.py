"""Unit tests for API client AI keyword detection, target roles, and search queries."""

from pathlib import Path
import pytest

from job_mcp.core.api_client import (
    CURATED_TECH_KEYWORDS,
    SPECIALIZED_COMPETENCIES,
    _derive_search_queries,
    _derive_target_roles,
    extract_candidate_profile,
    resolve_cv_path,
)
from job_mcp.models.schemas import CandidateProfile


def test_derive_genai_roles():
    """Verify that GenAI and Israeli market roles are derived when GenAI/AI skills are present."""
    skills = ["Python", "LangChain", "LangGraph", "RAG", "FastAPI", "Docker"]
    roles = _derive_target_roles(skills, "Built RAG systems with LLM APIs")
    assert "AI Engineer" in roles
    assert "GenAI Engineer" in roles
    assert "LLM Engineer" in roles
    assert "Algorithm Developer" in roles
    assert "Applied AI Engineer" in roles
    assert "AI Backend Developer" in roles


def test_derive_search_queries_includes_hebrew():
    """Verify that Hebrew queries are added when AI target roles are present."""
    top_skills = ["Python", "LangGraph", "RAG", "FastAPI"]
    target_roles = ["AI Engineer", "GenAI Engineer", "Backend Engineer"]
    queries = _derive_search_queries(top_skills, target_roles)
    assert any("מהנדס" in q or "אלגוריתמים" in q for q in queries)
    assert "מהנדס בינה מלאכותית" in queries
    assert "מפתח אלגוריתמים" in queries


def test_specialized_competencies_includes_genai_tools():
    """Verify that all new AI tools/keywords are present in SPECIALIZED_COMPETENCIES and CURATED_TECH_KEYWORDS."""
    expected_tools = [
        "openai", "gemini", "anthropic", "claude", "gpt", "fastmcp", "mlops",
        "triton", "onnx", "wandb", "mlflow", "milvus", "faiss", "lancedb",
    ]
    for tool in expected_tools:
        assert tool in SPECIALIZED_COMPETENCIES, f"{tool} missing from SPECIALIZED_COMPETENCIES"

    curated_lower = {k.lower() for k in CURATED_TECH_KEYWORDS}
    for tool in expected_tools:
        assert tool in curated_lower, f"{tool} missing from CURATED_TECH_KEYWORDS"


def test_extract_candidate_profile_cv_pdf_enriched_roles():
    """Verify that extracting candidate profile from cv.pdf outputs enriched target roles and Hebrew queries."""
    cv_path = Path("./cv.pdf")
    if not cv_path.exists():
        pytest.skip("cv.pdf not found in project root")

    profile = extract_candidate_profile(cv_path)
    assert "AI Engineer" in profile.target_roles
    assert "GenAI Engineer" in profile.target_roles
    assert "LLM Engineer" in profile.target_roles
    assert "Algorithm Developer" in profile.target_roles
    assert "Applied AI Engineer" in profile.target_roles
    assert "AI Backend Developer" in profile.target_roles
    assert any("מהנדס" in q or "אלגוריתמים" in q for q in profile.search_queries)


def test_extract_candidate_contact_info_from_cv():
    """Verify extracting candidate contact information dynamically from cv.pdf if present."""
    cv_path = Path("./cv.pdf")
    if not cv_path.exists():
        pytest.skip("cv.pdf not found in project root")

    profile = extract_candidate_profile(cv_path)
    assert profile.full_name is not None and len(profile.full_name.strip()) > 0
    assert profile.first_name is not None and len(profile.first_name.strip()) > 0
    assert profile.email is not None and "@" in profile.email


def test_candidate_contact_info_env_fallback(monkeypatch):
    """Verify contact info populates cleanly from CANDIDATE_* environment variables."""
    monkeypatch.setenv("CANDIDATE_NAME", "Jordan Blake")
    monkeypatch.setenv("CANDIDATE_FIRST_NAME", "Jordan")
    monkeypatch.setenv("CANDIDATE_LAST_NAME", "Blake")
    monkeypatch.setenv("CANDIDATE_EMAIL", "jordan@example.com")
    monkeypatch.setenv("CANDIDATE_PHONE", "+1-555-444-3322")
    monkeypatch.setenv("CANDIDATE_LINKEDIN", "https://linkedin.com/in/jordanblake")
    monkeypatch.setenv("CANDIDATE_GITHUB", "https://github.com/jordanblake")

    profile = extract_candidate_profile("")
    assert profile.full_name == "Jordan Blake"
    assert profile.first_name == "Jordan"
    assert profile.last_name == "Blake"
    assert profile.email == "jordan@example.com"
    assert profile.phone == "+1-555-444-3322"
    assert profile.linkedin_url == "https://www.linkedin.com/in/jordanblake"
    assert profile.github_url == "https://github.com/jordanblake"


def test_candidate_profile_contact_defaults():
    """Instantiating empty CandidateProfile() leaves contact fields as None."""
    profile = CandidateProfile()
    assert profile.full_name is None
    assert profile.first_name is None
    assert profile.last_name is None
    assert profile.email is None
    assert profile.phone is None
    assert profile.linkedin_url is None
    assert profile.github_url is None


def test_extract_candidate_contact_fallback_env(monkeypatch):
    """When raw text has no email/phone, falls back to env vars if present."""
    monkeypatch.setenv("CANDIDATE_NAME", "Jane Doe")
    monkeypatch.setenv("CANDIDATE_EMAIL", "jane.doe@example.com")
    monkeypatch.setenv("CANDIDATE_PHONE", "+1-555-123-4567")
    monkeypatch.setenv("CANDIDATE_LINKEDIN", "https://linkedin.com/in/janedoe")
    monkeypatch.setenv("CANDIDATE_GITHUB", "https://github.com/janedoe")

    raw_text = "Experienced Software Engineer specializing in Python, FastAPI, and Docker."
    profile = extract_candidate_profile(raw_text)

    assert profile.full_name == "Jane Doe"
    assert profile.first_name == "Jane"
    assert profile.last_name == "Doe"
    assert profile.email == "jane.doe@example.com"
    assert profile.phone == "+1-555-123-4567"
    assert profile.linkedin_url == "https://www.linkedin.com/in/janedoe"
    assert profile.github_url == "https://github.com/janedoe"


def test_extract_candidate_contact_info_from_raw_text(monkeypatch):
    """Verify inline regex extraction of contact info from structured raw text."""
    # Ensure env vars do not interfere
    for var in ("CANDIDATE_NAME", "CANDIDATE_EMAIL", "CANDIDATE_PHONE", "CANDIDATE_LINKEDIN", "CANDIDATE_GITHUB"):
        monkeypatch.delenv(var, raising=False)

    raw_text = """John Smith
Senior Backend Engineer
john.smith@domain.org | +1 (555) 987-6543 | linkedin.com/in/johnsmith | github.com/johnsmith
Skills: Python, Go, Docker
"""
    profile = extract_candidate_profile(raw_text)

    assert profile.full_name == "John Smith"
    assert profile.first_name == "John"
    assert profile.last_name == "Smith"
    assert profile.email == "john.smith@domain.org"
    assert profile.phone == "+1 (555) 987-6543"
    assert profile.linkedin_url == "https://www.linkedin.com/in/johnsmith"
    assert profile.github_url == "https://github.com/johnsmith"


def test_extract_candidate_name_with_apostrophe(monkeypatch):
    """Verify names with apostrophes (e.g. O'Connor, D'Angelo) are recognized."""
    for var in ("CANDIDATE_NAME", "CANDIDATE_EMAIL", "CANDIDATE_PHONE", "CANDIDATE_LINKEDIN", "CANDIDATE_GITHUB"):
        monkeypatch.delenv(var, raising=False)

    raw_text = """Liam O'Connor
Senior AI Engineer
liam@example.com | +972-50-9998877
"""
    profile = extract_candidate_profile(raw_text)
    assert profile.full_name == "Liam O'Connor"
    assert profile.first_name == "Liam"
    assert profile.last_name == "O'Connor"


def test_resolve_cv_path_generic(tmp_path, monkeypatch):
    """Verify resolve_cv_path discovers DEFAULT_CV_PATH or cv.pdf and does not require or mention personal filenames."""
    import inspect

    # Verify no personal filenames are mentioned in implementation or docstring
    source = inspect.getsource(resolve_cv_path)
    banned_filename = bytes.fromhex("6c696f725f7a7669656c69").decode()
    assert banned_filename not in source, "Personal filename found in resolve_cv_path source"

    # Test discovery via DEFAULT_CV_PATH environment variable
    custom_cv = tmp_path / "custom_candidate_resume.pdf"
    custom_cv.write_text("candidate resume content")
    monkeypatch.setenv("DEFAULT_CV_PATH", str(custom_cv))

    resolved = resolve_cv_path()
    assert resolved == custom_cv.resolve()

    # Test discovery via standard cv.pdf in cwd
    monkeypatch.delenv("DEFAULT_CV_PATH", raising=False)
    sub_dir = tmp_path / "workspace"
    sub_dir.mkdir()
    monkeypatch.chdir(sub_dir)
    std_cv = sub_dir / "cv.pdf"
    std_cv.write_text("standard cv content")

    resolved_std = resolve_cv_path()
    assert resolved_std == std_cv.resolve()


def test_resolve_default_cv_generic(tmp_path, monkeypatch):
    """Verify scripts/run_mock_llm_pipeline.py resolve_default_cv has no personal filenames."""
    import inspect
    from scripts.run_mock_llm_pipeline import resolve_default_cv

    source = inspect.getsource(resolve_default_cv)
    banned_filename = bytes.fromhex("6c696f725f7a7669656c69").decode()
    assert banned_filename not in source, "Personal filename found in resolve_default_cv source"
    assert "candidate_cv.pdf" in source

    # Verify discovery via candidate_cv.pdf when in cwd
    monkeypatch.delenv("DEFAULT_CV_PATH", raising=False)
    monkeypatch.delenv("CV_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    cand_cv = tmp_path / "candidate_cv.pdf"
    cand_cv.write_text("candidate cv content")

    resolved = resolve_default_cv()
    assert resolved == "candidate_cv.pdf"


def test_non_technical_roles_heavily_penalized_for_engineers():
    """Verify non-technical roles (Social Media, SDR, Recruiter) are disqualified (score <= 15) even if description has tech keywords."""
    from job_mcp.core.api_client import calculate_match_score
    from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences

    profile = CandidateProfile(
        target_roles=["Junior AI Engineer", "Backend Developer", "Software Engineer"],
        skills=["Python", "FastAPI", "SQL", "Docker", "Agentic", "C"],
        primary_stack=["Python", "FastAPI", "SQL"],
        top_skills=["Python", "FastAPI", "SQL"],
    )
    prefs = JobPreferences(tech_stack=["Python", "SQL"])

    # 1. Social Media Manager with boilerplate buzzwords (Port style)
    job_sm = Job(
        job_id="sm-1",
        title="Social Media Manager",
        company="Port",
        description="Port is the Agentic SDLC Platform. AI agents operating across the SDLC. C and Python developers.",
        tech_stack=["C", "Agentic", "Python"],
    )
    score_sm = calculate_match_score(job_sm, prefs, profile=profile)
    assert score_sm <= 15.0, f"Expected Social Media Manager score <= 15.0, got {score_sm}"

    # 2. Sales Development Representative (Incredibuild style)
    job_sdr = Job(
        job_id="sdr-1",
        title="Sales Development Representative",
        company="Incredibuild",
        description="Reach out to high volume outbound leads. Supporting agentic AI development needs.",
        tech_stack=["Agentic"],
    )
    score_sdr = calculate_match_score(job_sdr, prefs, profile=profile)
    assert score_sdr <= 15.0, f"Expected SDR score <= 15.0, got {score_sdr}"

    # 3. Talent Acquisition / Recruiter
    job_recruiter = Job(
        job_id="rec-1",
        title="Technical Talent Acquisition Specialist",
        company="TechTalent",
        description="Source and recruit top Python, Docker, and SQL engineers.",
        tech_stack=["Python", "Docker"],
    )
    score_rec = calculate_match_score(job_recruiter, prefs, profile=profile)
    assert score_rec <= 15.0, f"Expected Recruiter score <= 15.0, got {score_rec}"


def test_engineering_role_achieves_top_tier_score():
    """Verify genuine AI and backend engineering roles score >= 85 (Top-Tier)."""
    from job_mcp.core.api_client import calculate_match_score
    from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences

    profile = CandidateProfile(
        target_roles=["AI Engineer", "Backend Developer", "Software Engineer"],
        skills=["Python", "FastAPI", "SQL", "Docker", "LLM", "RAG"],
        primary_stack=["Python", "FastAPI", "LLM"],
        top_skills=["Python", "FastAPI", "LLM"],
    )
    prefs = JobPreferences(tech_stack=["Python", "FastAPI", "LLM"])

    job_ai = Job(
        job_id="ai-1",
        title="Junior AI Engineer",
        company="Claroty",
        description="Develop production AI systems using Python, FastAPI, Docker, and LLM RAG pipelines.",
        tech_stack=["Python", "FastAPI", "Docker", "LLM", "RAG"],
    )
    score_ai = calculate_match_score(job_ai, prefs, profile=profile)
    assert score_ai >= 85.0, f"Expected AI Engineer score >= 85.0, got {score_ai}"


def test_unmatched_target_role_capped_at_strong_match():
    """Verify tech jobs without matching target role (e.g. Escalation Engineer) do not exceed 75.0 (never 88.0)."""
    from job_mcp.core.api_client import calculate_match_score
    from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences

    profile = CandidateProfile(
        target_roles=["Junior AI Engineer", "Backend Developer"],
        skills=["Python", "SQL", "REST"],
        primary_stack=["Python"],
        top_skills=["Python"],
    )
    prefs = JobPreferences(tech_stack=["Python"])

    job_esc = Job(
        job_id="esc-1",
        title="Escalation Engineer",
        company="Rapyd",
        description="Fintech platform operations with SQL and REST APIs.",
        tech_stack=["SQL", "REST"],
    )
    score_esc = calculate_match_score(job_esc, prefs, profile=profile)
    assert score_esc <= 75.0, f"Expected unmatched target role score <= 75.0, got {score_esc}"



