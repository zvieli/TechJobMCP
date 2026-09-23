"""Unit tests for LazyLayaEngine lifecycle, lazy loading, and fallback handling."""

from unittest.mock import MagicMock, patch
import pytest

from job_mcp.core.system1.engine import LazyLayaEngine


class TestLazyLayaEngine:
    def setup_method(self):
        LazyLayaEngine.reset_instance()

    def teardown_method(self):
        LazyLayaEngine.reset_instance()

    def test_singleton_behavior(self):
        e1 = LazyLayaEngine.get_instance()
        e2 = LazyLayaEngine.get_instance()
        assert e1 is e2
        assert not e1.is_loaded()

    def test_fallback_when_model_missing(self):
        engine = LazyLayaEngine(model_name="data/models/nonexistent", auto_release_after_batch=False)
        score, conf = engine.predict_score("state", "question", ["A", "B", "C"])
        assert score == 2
        assert conf == 0.50

        prob = engine.predict_noul("state", "question")
        assert prob == 0.50

        choice, c_conf = engine.predict_choice("state", "question", ["Option1", "Option2"])
        assert choice == "Option1"

    def test_match_scoring_ensemble_execution(self):
        engine = LazyLayaEngine(auto_release_after_batch=True)
        res = engine.predict_match_scoring_ensemble(
            job_desc="Python Backend Engineer, 5+ yrs experience in FastAPI and AWS",
            cv_text="Software Developer, 6 yrs experience in Python, FastAPI, and AWS cloud",
        )
        assert "skill_match" in res
        assert "seniority_fit" in res
        assert "recruiter_fit_probability" in res
        assert 0 <= res["skill_match"] <= 4
        assert 0.0 <= res["skill_confidence"] <= 1.0
        assert 0 <= res["seniority_fit"] <= 4
        assert 0.0 <= res["seniority_confidence"] <= 1.0
        assert 0.0 <= res["recruiter_fit_probability"] <= 1.0
        assert not engine.is_loaded()  # Auto-released
