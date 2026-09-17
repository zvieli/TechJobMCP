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
