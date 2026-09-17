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
    assert "lior_zvieli" not in source, "Personal filename found in resolve_cv_path source"

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
    assert "lior_zvieli" not in source, "Personal filename found in resolve_default_cv source"
    assert "candidate_cv.pdf" in source

    # Verify discovery via candidate_cv.pdf when in cwd
    monkeypatch.delenv("DEFAULT_CV_PATH", raising=False)
    monkeypatch.delenv("CV_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    cand_cv = tmp_path / "candidate_cv.pdf"
    cand_cv.write_text("candidate cv content")

    resolved = resolve_default_cv()
    assert resolved == "candidate_cv.pdf"


