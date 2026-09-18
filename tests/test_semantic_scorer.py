"""Unit tests for CPU-based SemanticScorer using fastembed ONNX."""

import concurrent.futures
from unittest.mock import MagicMock, patch
import pytest

from job_mcp.core.semantic_scorer import SemanticScorer


def test_is_available():
    """Verify fastembed availability detection."""
    available = SemanticScorer.is_available()
    assert isinstance(available, bool)
    scorer = SemanticScorer.get_instance()
    assert scorer.is_available() == available


@pytest.mark.skipif(not SemanticScorer.is_available(), reason="fastembed/numpy is an optional dependency")
def test_score_similarity_live():
    """Verify real semantic similarity computation using fastembed ONNX model."""
    scorer = SemanticScorer.get_instance()
    query = "Senior Python Backend Engineer with FastAPI and PostgreSQL"
    doc_relevant = "We are looking for a Python Developer experienced with FastAPI, SQL databases and microservices."
    doc_unrelated = "Looking for an Executive Chef to oversee our Mediterranean kitchen and manage food supplies."

    scores = scorer.score_similarity(query, [doc_relevant, doc_unrelated])

    assert len(scores) == 2
    assert scores[0] > scores[1], f"Expected relevant score ({scores[0]}) > unrelated score ({scores[1]})"
    assert scores[0] > 0.60, f"Expected relevant score ({scores[0]}) > 0.60"
    assert 0.0 <= scores[1] <= 1.0

    # Also test score_single
    single_score = scorer.score_single(query, doc_relevant)
    assert pytest.approx(single_score, abs=1e-4) == scores[0]


def test_fallback_when_unavailable():
    """Verify graceful fallback returning zeros when fastembed is unavailable."""
    with patch.object(SemanticScorer, "is_available", return_value=False):
        scorer = SemanticScorer()
        scores = scorer.score_similarity(
            "Python Engineer",
            ["Software Developer", "Chef"],
        )
        assert scores == [0.0, 0.0]
        assert scorer.score_single("Python Engineer", "Software Developer") == 0.0


def test_error_handling_graceful():
    """Verify graceful zero return when embedding model raises an error."""
    scorer = SemanticScorer()
    with patch.object(scorer, "_get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.embed.side_effect = RuntimeError("ONNX Runtime execution failed")
        mock_get_model.return_value = mock_model

        scores = scorer.score_similarity(
            "Python Engineer",
            ["Software Developer", "Data Scientist"],
        )
        assert scores == [0.0, 0.0]
        assert scorer.score_single("Python Engineer", "Software Developer") == 0.0


def test_empty_inputs():
    """Verify handling of empty query, empty documents list, and whitespace-only strings."""
    scorer = SemanticScorer.get_instance()

    # Empty documents list
    assert scorer.score_similarity("Python", []) == []

    # Empty or whitespace query
    assert scorer.score_similarity("", ["Doc 1", "Doc 2"]) == [0.0, 0.0]
    assert scorer.score_similarity("   \t\n  ", ["Doc 1", "Doc 2"]) == [0.0, 0.0]

    # Empty or whitespace documents in the list
    scores = scorer.score_similarity("Python", ["", "   ", "Python Developer"])
    assert len(scores) == 3
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    if scorer.is_available():
        assert scores[2] > 0.0
    else:
        assert scores[2] == 0.0

    # Empty score_single
    assert scorer.score_single("", "Doc") == 0.0
    assert scorer.score_single("Query", "") == 0.0
    assert scorer.score_single("   ", "   ") == 0.0


def test_singleton_thread_safety():
    """Verify thread-safe singleton initialization and consistency."""
    # Reset singleton instance for testing clean creation
    SemanticScorer.reset_instance()

    instances = []

    def fetch_instance():
        return SemanticScorer.get_instance()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_instance) for _ in range(20)]
        for f in concurrent.futures.as_completed(futures):
            instances.append(f.result())

    first = instances[0]
    assert all(inst is first for inst in instances)
