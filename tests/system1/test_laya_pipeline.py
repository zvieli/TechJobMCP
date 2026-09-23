"""Tests for Laya dataset preparation, calibration, and training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_laya_dataset import (
    convert_dedup_verification,
    convert_field_mapping,
    convert_match_scoring,
    convert_role_classification,
    convert_section_parsing,
)
from scripts.train_laya import compute_ece, grid_search_temperature


def test_convert_match_scoring():
    sample = {
        "job_title": "DevOps Engineer",
        "job_description": "Kubernetes, CI/CD, AWS",
        "candidate_cv": "10 years DevOps and AWS",
        "skill_match": 4,
        "seniority_fit": 3,
        "recruiter_fit": True,
    }
    items = convert_match_scoring(sample)
    assert len(items) == 3

    skill, sen, rec = items[0], items[1], items[2]
    assert skill["task"] == "match_scoring_skill"
    assert skill["qtype"] == "score"
    assert skill["label"] == 4
    assert len(skill["options"]) == 5

    assert sen["task"] == "match_scoring_seniority"
    assert sen["label"] == 3

    assert rec["task"] == "match_scoring_recruiter_fit"
    assert rec["qtype"] == "noul"
    assert rec["label"] == 1


def test_convert_dedup_verification():
    sample = {
        "job_a": {"title": "DevOps", "company": "Co", "location": "TLV"},
        "job_b": {"title": "DevOps", "company": "Co", "location": "TLV"},
        "is_duplicate": True,
        "reasoning": "identical",
    }
    items = convert_dedup_verification(sample)
    assert len(items) == 1
    assert items[0]["task"] == "dedup_verification"
    assert items[0]["label"] == 1


def test_convert_role_classification():
    sample = {
        "title": "Software Engineer",
        "snippet": "Python backend development",
        "category": "Core Engineering",
    }
    items = convert_role_classification(sample)
    assert len(items) == 1
    assert items[0]["task"] == "role_classification"
    assert items[0]["label"] == 0  # Core Engineering is index 0


def test_convert_section_parsing():
    sample = {
        "paragraph": "Requirements: 5 years experience.",
        "section_type": "Requirements",
    }
    items = convert_section_parsing(sample)
    assert len(items) == 1
    assert items[0]["task"] == "section_parsing"
    assert items[0]["label"] == 0  # Requirements is index 0


def test_convert_field_mapping():
    sample = {
        "label": "Email Address",
        "placeholder": "name@example.com",
        "field_type": "email",
        "field_intent": "Email",
    }
    items = convert_field_mapping(sample)
    assert len(items) == 1
    assert items[0]["task"] == "field_mapping"
    assert items[0]["label"] == 3  # Email is index 3


def test_ece_computation():
    # Well-calibrated predictions: confidence is max probability
    probs = [0.95, 0.90, 0.92, 0.88]
    preds = [1, 1, 0, 0]
    targets = [1, 1, 0, 0]
    ece = compute_ece(probs, preds, targets, n_bins=5)
    assert 0.0 <= ece <= 0.15


def test_grid_search_temperature():
    # Synthetic logits where scale is slightly overconfident
    logits = [
        [10.0, 1.0],
        [8.0, 2.0],
        [1.0, 9.0],
        [2.0, 8.0],
    ]
    targets = [0, 0, 1, 1]
    best_t, calibrated_ece, uncal_ece = grid_search_temperature(logits, targets)
    assert 0.1 <= best_t <= 5.0
    assert calibrated_ece >= 0.0
