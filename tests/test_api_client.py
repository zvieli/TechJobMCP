"""Unit tests for API client AI keyword detection, target roles, and search queries."""

from pathlib import Path
import pytest

from job_mcp.core.api_client import (
    CURATED_TECH_KEYWORDS,
    SPECIALIZED_COMPETENCIES,
    _derive_search_queries,
    _derive_target_roles,
    extract_candidate_profile,
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
    """Verify extracting candidate contact information from cv.pdf."""
    cv_path = Path("./cv.pdf")
    if not cv_path.exists():
        pytest.skip("cv.pdf not found in project root")

    profile = extract_candidate_profile(cv_path)
    assert profile.full_name == "Lior Zvieli"
    assert profile.first_name == "Lior"
    assert profile.last_name == "Zvieli"
    assert profile.email == "liorzvieli@gmail.com"
    assert profile.phone == "+972-52-2276810"
    assert profile.github_url == "https://github.com/zvieli"


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
    assert profile.linkedin_url == "https://linkedin.com/in/janedoe"
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

