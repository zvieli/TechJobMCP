"""Unit tests for System 1 Abstraction Layer and Dependency Injection."""

from typing import Any, Dict, List, Tuple
import pytest

from job_mcp.core.api_client import calculate_match_score
from job_mcp.core.system1.engine import LazyLayaEngine
from job_mcp.core.system1.factory import get_system1_engine, set_active_engine
from job_mcp.core.system1.generative import GenerativeBaselineEngine
from job_mcp.core.system1.interface import System1Engine
from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences, WorkMode


class MockSystem1Engine:
    """Mock engine implementing System1Engine for testing dependency injection."""

    def __init__(self, skill_match: int = 4, seniority_fit: int = 2, rec_prob: float = 0.95):
        self.skill_match = skill_match
        self.seniority_fit = seniority_fit
        self.rec_prob = rec_prob
        self.calls: List[str] = []

    def is_loaded(self) -> bool:
        return True

    def predict_score(self, state: str, question: str, options: List[str]) -> Tuple[int, float]:
        self.calls.append("predict_score")
        return self.skill_match, 0.90

    def predict_noul(self, state: str, question: str) -> float:
        self.calls.append("predict_noul")
        return self.rec_prob

    def predict_choice(self, state: str, question: str, options: List[str]) -> Tuple[str, float]:
        self.calls.append("predict_choice")
        idx = min(self.skill_match, len(options) - 1)
        return options[idx], 0.90

    def predict_match_scoring_ensemble(
        self, job_desc: str = "", cv_text: str = "", job_title: str = "Unknown Role"
    ) -> Dict[str, Any]:
        self.calls.append(f"ensemble:{job_title}")
        return {
            "skill_match": self.skill_match,
            "skill_confidence": 0.92,
            "seniority_fit": self.seniority_fit,
            "seniority_confidence": 0.94,
            "recruiter_fit_probability": self.rec_prob,
        }

    def predict_match_scoring_batch(
        self, items: List[Dict[str, str]], chunk_size: int = 4
    ) -> List[Dict[str, Any]]:
        self.calls.append(f"batch:{len(items)}")
        return [
            {
                "skill_match": self.skill_match,
                "skill_confidence": 0.92,
                "seniority_fit": self.seniority_fit,
                "seniority_confidence": 0.94,
                "recruiter_fit_probability": self.rec_prob,
            }
            for _ in items
        ]


def test_default_engine_is_lazylaya():
    """Verify that get_system1_engine defaults to the LazyLayaEngine singleton."""
    set_active_engine(None)
    engine = get_system1_engine()
    assert isinstance(engine, LazyLayaEngine)
    assert engine is LazyLayaEngine.get_instance()


def test_dependency_injection_custom_engine():
    """Verify that set_active_engine successfully swaps the System 1 decision engine."""
    mock_engine = MockSystem1Engine(skill_match=4, seniority_fit=2, rec_prob=0.99)
    set_active_engine(mock_engine)

    try:
        active = get_system1_engine()
        assert active is mock_engine

        job = Job(
            job_id="test_inj_1",
            title="Senior Python Architect",
            company="TestCo",
            location="Remote",
            work_mode=WorkMode.REMOTE,
            tech_stack=["Python", "FastAPI"],
            description="Build distributed systems in Python.",
        )
        profile = CandidateProfile(
            skills=["Python", "FastAPI"],
            top_skills=["Python", "FastAPI"],
            primary_stack=["Python", "FastAPI"],
            seniority_level="Senior",
            years_of_experience=6,
        )
        prefs = JobPreferences(cv_path="mock_cv")

        score = calculate_match_score(job, prefs, profile=profile, enable_system1=True)

        assert any(c.startswith("ensemble:Senior Python Architect") for c in mock_engine.calls)
        assert score >= 80.0
        assert job.system1_confidence is not None
        assert job.system1_confidence >= 0.85
    finally:
        set_active_engine(None)


def test_generative_baseline_interface_conformance():
    """Verify that GenerativeBaselineEngine adheres to System1Engine Protocol."""
    gen_engine = GenerativeBaselineEngine()
    assert isinstance(gen_engine, System1Engine)
    assert gen_engine.is_loaded() is True
