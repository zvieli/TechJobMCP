"""Tests for Base (pre-trained, non-fine-tuned) LAYA Engine."""

import pytest
from job_mcp.core.system1.interface import System1Engine
from job_mcp.core.system1.factory import set_active_engine, get_system1_engine, reset_engine
from job_mcp.core.system1.base_laya import BaseLayaEngine


def test_base_laya_satisfies_protocol():
    """Verify that BaseLayaEngine conforms to System1Engine Protocol."""
    engine = BaseLayaEngine()
    assert isinstance(engine, System1Engine)


def test_base_laya_factory_injection():
    """Verify that BaseLayaEngine can be injected into the factory."""
    engine = BaseLayaEngine()
    set_active_engine(engine)
    try:
        assert get_system1_engine() is engine
    finally:
        reset_engine()


def test_base_laya_predict_ensemble():
    """Verify ensemble prediction structure."""
    engine = BaseLayaEngine()
    cv_text = "Skills: Python, React, Docker. Seniority: Junior."
    job_desc = "Junior Software Engineer with Python and Docker."
    
    result = engine.predict_match_scoring_ensemble(job_desc=job_desc, cv_text=cv_text, job_title="Junior Developer")
    assert "skill_match" in result
    assert "skill_confidence" in result
    assert "seniority_fit" in result
    assert "seniority_confidence" in result
    assert "recruiter_fit_probability" in result

    assert 0 <= result["skill_match"] <= 4
    assert 0 <= result["seniority_fit"] <= 2
    assert 0.0 <= result["recruiter_fit_probability"] <= 1.0


def test_base_laya_predict_batch():
    """Verify batched prediction structure."""
    engine = BaseLayaEngine()
    items = [
        {"job_desc": "Junior Python Dev", "cv_text": "Junior Python Dev"},
        {"job_desc": "Senior Architect", "cv_text": "Junior Python Dev"},
    ]
    batch_results = engine.predict_match_scoring_batch(items)
    assert len(batch_results) == 2
    for r in batch_results:
        assert "skill_match" in r
        assert "seniority_fit" in r
        assert "recruiter_fit_probability" in r
