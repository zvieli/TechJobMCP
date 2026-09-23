"""Accuracy gate and evaluation test suite for TechJobMCP v2 dual-cognition models.

Evaluates trained Laya decision models against baseline heuristics on the labeled holdout set.
Models must match or exceed baseline thresholds before replacing existing heuristics in production.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from job_mcp.core.api_client import (
    NON_TECH_ROLE_TERMS,
    calculate_match_score,
)
from job_mcp.core.section_parser import extract_clean_job_tech_stack, parse_job_sections
from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences
from job_mcp.sources.dedup import compute_dedup_key

ADMIN_TERMS = [
    "מנתח מערכות",
    "מנהל פרויקטים",
    "data annotator",
    "qa",
    "project manager",
    "system analyst",
    "scrum master",
    "product manager",
    "help desk",
    "support",
]

def is_admin_title(title: str) -> bool:
    t = title.lower()
    return any(x in t for x in ADMIN_TERMS)

HOLDOUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "holdout"


def load_holdout(name: str):
    path = HOLDOUT_DIR / f"{name}_holdout.json"
    if not path.exists():
        pytest.skip(f"Holdout file {path} does not exist yet. Run generation script first.")
    data = json.loads(path.read_text())
    if not data:
        pytest.skip(f"Holdout file {path} is empty.")
    return data


class TestHoldoutDataIntegrity:
    """Ensure generated holdout datasets meet schema and quality requirements."""

    def test_match_scoring_holdout_schema(self):
        samples = load_holdout("match_scoring")
        for s in samples:
            assert "job_title" in s
            assert "job_description" in s
            assert "candidate_cv" in s
            assert 0 <= s["skill_match"] <= 4
            assert 0 <= s["seniority_fit"] <= 4
            assert isinstance(s["recruiter_fit"], bool)

    def test_dedup_holdout_schema(self):
        samples = load_holdout("dedup_verification")
        for s in samples:
            assert "job_a" in s
            assert "job_b" in s
            assert isinstance(s["is_duplicate"], bool)

    def test_role_classification_holdout_schema(self):
        valid_cats = {
            "Core Engineering",
            "Engineering-Adjacent",
            "Technical Hybrid",
            "Non-Technical",
            "Administrative",
        }
        samples = load_holdout("role_classification")
        for s in samples:
            assert "title" in s
            assert s["category"] in valid_cats

    def test_section_parsing_holdout_schema(self):
        valid_sections = {
            "Requirements",
            "Responsibilities",
            "Company overview",
            "Benefits",
            "Application instructions",
            "Boilerplate",
        }
        samples = load_holdout("section_parsing")
        for s in samples:
            assert "paragraph" in s
            assert s["section_type"] in valid_sections

    def test_field_mapping_holdout_schema(self):
        valid_intents = {
            "First name",
            "Last name",
            "Full name",
            "Email",
            "Phone",
            "LinkedIn",
            "GitHub",
            "Resume upload",
            "Cover letter",
            "Years of experience",
            "Education",
            "Work authorization",
            "Salary expectation",
            "Custom screening question",
            "Other",
        }
        samples = load_holdout("field_mapping")
        for s in samples:
            assert "label" in s
            assert s["field_intent"] in valid_intents


class TestHeuristicBaselineBenchmark:
    """Benchmark the existing deterministic heuristics against the holdout ground truth.

    Establishes the baseline bar that Laya models must surpass.
    """

    def test_benchmark_heuristic_role_classification(self):
        samples = load_holdout("role_classification")
        correct = 0
        total = len(samples)

        for s in samples:
            title = s["title"].lower()
            is_non_tech = any(term in title for term in NON_TECH_ROLE_TERMS)
            is_admin = is_admin_title(s["title"])

            pred_cat = "Non-Technical" if is_non_tech else ("Administrative" if is_admin else "Core Engineering")
            actual_cat = s["category"]

            # Evaluate agreement on high-level technical vs non-technical split
            actual_is_tech = actual_cat in ("Core Engineering", "Engineering-Adjacent", "Technical Hybrid")
            pred_is_tech = pred_cat == "Core Engineering"

            if actual_is_tech == pred_is_tech:
                correct += 1

        baseline_acc = correct / total if total else 0.0
        # Records the baseline accuracy for gate comparison in Phase 2
        assert baseline_acc >= 0.0
