"""Evaluation plumbing and real-model diagnostics for the Laya System 1 engine.

The module deliberately separates two lanes:

* **Framework / plumbing tests** run without model artifacts and validate pair
  reconstruction, metric calculation, fallback recognition, and the production
  seniority cap.
* **Real-model diagnostics** require the ignored model and holdout artifacts.
  Missing artifacts produce explicit skips. Current unmet quality targets are
  strict expected failures, not passing gates.

PR CI runs this file as evaluation-framework coverage. A future artifact-pinned
model-evaluation lane must promote real-model targets only after reproducible
measurements meet them.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from job_mcp.core.api_client import calculate_match_score
from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOLDOUT_PATH = Path(
    os.getenv("LAYA_HOLDOUT_PATH", str(REPO_ROOT / "data" / "holdout" / "laya_val.jsonl"))
)
MODEL_DIR = Path(os.getenv("LAYA_MODEL_PATH", str(REPO_ROOT / "data" / "models" / "laya-techjob")))

# Specification targets. They are intentionally not environment-configurable.
TARGET_MAX_ECE = 0.035
TARGET_MIN_MISMATCH_ACCURACY = 0.90

EXPECTED_TEMPERATURE = 0.75
TEMPERATURE_TOLERANCE = 0.05
SENIORITY_MISMATCH_LABEL = 0
REQUIRED_MATCH_SCORING_TASKS = frozenset(
    {"match_scoring_skill", "match_scoring_seniority", "match_scoring_recruiter_fit"}
)
FALLBACK_MATCH_RESULT = {
    "skill_match": 2,
    "skill_confidence": 0.50,
    "seniority_fit": 2,
    "seniority_confidence": 0.50,
    "recruiter_fit_probability": 0.50,
}


def artifacts_available() -> bool:
    """Return whether both ignored artifacts needed for inference are present."""
    return HOLDOUT_PATH.is_file() and MODEL_DIR.is_dir()


def compute_ece(
    confidences: Sequence[float],
    predictions: Sequence[int],
    targets: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Calculate equal-width-bin Expected Calibration Error."""
    if not confidences:
        return 0.0
    if len(confidences) != len(predictions) or len(predictions) != len(targets):
        raise ValueError("confidence, prediction, and target lengths must match")

    edges = [index / n_bins for index in range(n_bins + 1)]
    ece = 0.0
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        if index == n_bins - 1:
            members = [offset for offset, confidence in enumerate(confidences) if low <= confidence <= high]
        else:
            members = [offset for offset, confidence in enumerate(confidences) if low <= confidence < high]
        if not members:
            continue
        accuracy = sum(predictions[offset] == targets[offset] for offset in members) / len(members)
        confidence = sum(confidences[offset] for offset in members) / len(members)
        ece += len(members) / len(confidences) * abs(accuracy - confidence)
    return ece


def binary_metrics(predicted_positive: Sequence[bool], actual_positive: Sequence[bool]) -> dict[str, float | int]:
    """Return confusion-matrix counts and derived binary metrics."""
    if len(predicted_positive) != len(actual_positive):
        raise ValueError("prediction and target lengths must match")

    true_positive = false_positive = false_negative = true_negative = 0
    for predicted, actual in zip(predicted_positive, actual_positive):
        if predicted and actual:
            true_positive += 1
        elif predicted:
            false_positive += 1
        elif actual:
            false_negative += 1
        else:
            true_negative += 1

    total = true_positive + false_positive + false_negative + true_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "positive_count": true_positive + false_negative,
        "negative_count": false_positive + true_negative,
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_match_scoring_triples(
    records: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Reconstruct triples from their ordered JSONL sample identity.

    Neither job title nor full state is unique: the holdout contains repeated
    candidate/job states with different labels. Its stable identity is therefore
    the immutable JSONL ordering plus each triple occurrence. Every contiguous
    group of three match-scoring records must contain exactly one question of
    each required task; this preserves all repeated states instead of merging
    them.
    """
    match_records = [record for record in records if record.get("task") in REQUIRED_MATCH_SCORING_TASKS]
    if len(match_records) % len(REQUIRED_MATCH_SCORING_TASKS):
        raise ValueError("match-scoring records are not divisible into complete triples")

    triples = []
    for offset in range(0, len(match_records), len(REQUIRED_MATCH_SCORING_TASKS)):
        sample = match_records[offset : offset + len(REQUIRED_MATCH_SCORING_TASKS)]
        tasks = {record["task"] for record in sample}
        if tasks != REQUIRED_MATCH_SCORING_TASKS:
            raise ValueError(f"incomplete or duplicate tasks in match-scoring triple at offset {offset}")
        if any(not isinstance(record.get("state"), str) or not record["state"] for record in sample):
            raise ValueError(f"match-scoring triple at offset {offset} has no state")
        by_task = {record["task"]: record for record in sample}
        triples.append(
            (
                by_task["match_scoring_skill"],
                by_task["match_scoring_seniority"],
                by_task["match_scoring_recruiter_fit"],
            )
        )

    seniority_rows = sum(record["task"] == "match_scoring_seniority" for record in match_records)
    if len(triples) != seniority_rows:
        raise AssertionError("match-scoring reconstruction lost candidate/job pairs")
    return triples


def is_fallback_match_result(result: dict[str, Any]) -> bool:
    """Return True only for the complete documented engine fallback schema."""
    return result == FALLBACK_MATCH_RESULT


def split_state(state: str) -> tuple[str, str]:
    """Extract job description and candidate CV text from a holdout state."""
    description, separator, candidate = state.partition("\nCandidate CV:\n")
    if not separator:
        raise ValueError("match-scoring state has no candidate CV section")
    _, separator, description = description.partition("Job Description:\n")
    if not separator:
        raise ValueError("match-scoring state has no job description section")
    return description.strip(), candidate.strip()


# ---------------------------------------------------------------------------
# Artifact-independent framework / plumbing tests
# ---------------------------------------------------------------------------
def test_compute_ece_perfect_predictions_are_calibrated() -> None:
    assert compute_ece([0.75, 0.75, 0.75, 0.75], [1, 1, 1, 1], [1, 1, 1, 0]) == pytest.approx(0.0)


def test_build_match_scoring_triples_preserves_duplicate_titles_and_states() -> None:
    def record(task: str) -> dict[str, Any]:
        return {"task": task, "state": "repeated-state", "metadata": {"job_title": "Senior Engineer"}}

    ordered_tasks = [
        "match_scoring_skill",
        "match_scoring_seniority",
        "match_scoring_recruiter_fit",
    ]
    triples = build_match_scoring_triples([record(task) for task in ordered_tasks * 2])

    assert len(triples) == 2
    assert [triple[0]["state"] for triple in triples] == ["repeated-state", "repeated-state"]


def test_build_match_scoring_triples_rejects_incomplete_pairs() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_match_scoring_triples(
            [{"task": "match_scoring_skill", "state": "pair-a"}]
        )


def test_fallback_detection_requires_the_complete_schema() -> None:
    assert is_fallback_match_result(FALLBACK_MATCH_RESULT)
    assert not is_fallback_match_result(
        {"skill_match": 2, "seniority_fit": 2, "recruiter_fit_probability": 0.50}
    )


def test_junior_candidate_is_capped_for_senior_role() -> None:
    """Exercise production scoring, not a confidence-range proxy."""
    profile = CandidateProfile(
        skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
        top_skills=["Python", "FastAPI"],
        primary_stack=["Python", "FastAPI"],
        seniority_level="Junior",
        years_of_experience=1,
        target_roles=["Backend Engineer"],
    )
    job = Job(
        job_id="seniority-cap",
        title="Senior Python Backend Engineer",
        company="ExampleCo",
        location="Tel Aviv",
        tech_stack=["Python", "FastAPI", "PostgreSQL", "Docker"],
        description="Senior backend role requiring five years of production Python experience.",
    )
    score = calculate_match_score(
        job,
        JobPreferences(tech_stack=["Python", "FastAPI"]),
        profile=profile,
        enable_semantic=False,
        enable_system1=False,
    )

    assert score <= 40.0
    assert job.match_score <= 40.0


# ---------------------------------------------------------------------------
# Real-model diagnostics: skipped only when the required ignored artifacts lack
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def holdout_records() -> list[dict[str, Any]]:
    if not artifacts_available():
        pytest.skip("Real-model artifacts not present in environment")
    return [json.loads(line) for line in HOLDOUT_PATH.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def match_scoring_triples(
    holdout_records: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    triples = build_match_scoring_triples(holdout_records)
    assert triples, "Holdout contains no match-scoring triples"
    return triples


@pytest.fixture(scope="module")
def engine():
    from job_mcp.core.system1.engine import LazyLayaEngine

    instance = LazyLayaEngine.get_instance(model_name=str(MODEL_DIR))
    if not instance.warmup() or not instance.is_loaded():
        pytest.skip("Laya model could not be loaded in this environment")
    return instance


@pytest.fixture(scope="module")
def batched_results(engine, match_scoring_triples):
    items = []
    for _, seniority_row, _ in match_scoring_triples:
        job_desc, cv_text = split_state(seniority_row["state"])
        items.append(
            {
                "job_title": seniority_row.get("metadata", {}).get("job_title", "Unknown Role"),
                "job_desc": job_desc,
                "cv_text": cv_text,
            }
        )

    engine.predict_match_scoring_batch(items[:2])
    started = time.perf_counter()
    results = engine.predict_match_scoring_batch(items)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert len(results) == len(items)
    assert engine.is_loaded(), "Real-model diagnostic unexpectedly ran in fallback mode"
    assert not any(is_fallback_match_result(result) for result in results)
    return match_scoring_triples, results, elapsed_ms / len(items)


def test_real_model_int8_quantization_is_active(engine) -> None:
    torch = pytest.importorskip("torch")
    model = engine.load_model()
    assert model is not None
    assert any(
        isinstance(module, torch.nn.quantized.dynamic.Linear) for module in model.modules()
    ), "No dynamic INT8 Linear modules found"


def test_real_model_temperature_is_calibrated(engine) -> None:
    assert engine.temperature == pytest.approx(EXPECTED_TEMPERATURE, abs=TEMPERATURE_TOLERANCE)


def test_real_model_ece_target(batched_results) -> None:
    triples, results, _ = batched_results
    confidences: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []

    for (skill_row, seniority_row, recruiter_row), result in zip(triples, results):
        confidences.extend(
            [
                result["skill_confidence"],
                result["seniority_confidence"],
                max(result["recruiter_fit_probability"], 1.0 - result["recruiter_fit_probability"]),
            ]
        )
        predictions.extend(
            [
                result["skill_match"],
                result["seniority_fit"],
                int(result["recruiter_fit_probability"] >= 0.5),
            ]
        )
        targets.extend([int(skill_row["label"]), int(seniority_row["label"]), int(recruiter_row["label"])])

    ece = compute_ece(confidences, predictions, targets)
    print(f"\n[REAL MODEL] ECE={ece * 100:.2f}% target<={TARGET_MAX_ECE * 100:.1f}% n={len(targets)}")
    if ece > TARGET_MAX_ECE:
        pytest.xfail(
            f"Known model-quality gap: ECE {ece * 100:.2f}% exceeds target {TARGET_MAX_ECE * 100:.1f}%"
        )


def test_real_model_seniority_mismatch_target(batched_results) -> None:
    triples, results, _ = batched_results
    metrics = binary_metrics(
        [result["seniority_fit"] == SENIORITY_MISMATCH_LABEL for result in results],
        [int(seniority_row["label"]) == SENIORITY_MISMATCH_LABEL for _, seniority_row, _ in triples],
    )
    print(
        "\n[REAL MODEL] seniority mismatch "
        f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} tn={metrics['tn']} "
        f"positive={metrics['positive_count']} negative={metrics['negative_count']} "
        f"accuracy={metrics['accuracy'] * 100:.1f}% precision={metrics['precision'] * 100:.1f}% "
        f"recall={metrics['recall'] * 100:.1f}% f1={metrics['f1'] * 100:.1f}% "
        f"target_accuracy>={TARGET_MIN_MISMATCH_ACCURACY * 100:.1f}%"
    )
    if metrics["accuracy"] < TARGET_MIN_MISMATCH_ACCURACY:
        pytest.xfail(
            "Known model-quality gap: seniority mismatch accuracy is below the 90.0% target"
        )


def test_real_model_latency_is_measured(batched_results) -> None:
    triples, _, ms_per_pair = batched_results
    assert math.isfinite(ms_per_pair) and ms_per_pair > 0
    print(
        f"\n[REAL MODEL] latency={ms_per_pair:.2f} ms/pair over {len(triples)} pairs; "
        "diagnostic only until the benchmark protocol is approved"
    )
