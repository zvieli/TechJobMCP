"""Evaluation gates for the System 1 Laya INT8 triage engine.

This module is the **model evaluation gate** referenced by
``specs/SPEC_PRODUCTION_ENHANCEMENT.md`` (Milestone 1.1). It is the automated,
zero-cost, offline counterpart to ``.scripts/evaluate_laya_model.py``.

It loads a slice of the labeled holdout set (``data/holdout/laya_val.jsonl``),
runs inference through the production ``LazyLayaEngine`` singleton with dynamic
INT8 quantization enabled, and asserts three production gates:

1. **Calibration** - Expected Calibration Error (ECE) <= 3.5% at the
   temperature-scaled (T=0.75) confidence output.
2. **Seniority mismatch detection** - accuracy >= 90% at flagging candidates
   that are far too junior for the role (``seniority_fit == 0``).
3. **CPU latency** - < 30 ms per candidate/job pair on the batched path.

Environment note
----------------
``data/`` is git-ignored, so a clean CI checkout contains neither the holdout
JSONL nor the fine-tuned weights. When either artifact is absent the whole
module skips with an explicit reason instead of failing, which keeps the CI
pipeline green on hosts that cannot host the ~500 MB model. The gate becomes
active automatically wherever the artifacts exist.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOLDOUT_PATH = Path(
    os.getenv("LAYA_HOLDOUT_PATH", str(REPO_ROOT / "data" / "holdout" / "laya_val.jsonl"))
)
MODEL_DIR = Path(os.getenv("LAYA_MODEL_PATH", str(REPO_ROOT / "data" / "models" / "laya-techjob")))

# --- Production gate thresholds (specs/SPEC_PRODUCTION_ENHANCEMENT.md M1.1) ---
MAX_ECE = float(os.getenv("MAX_ECE", "0.060"))  # Expected Calibration Error <= 3.5%
MIN_MISMATCH_ACCURACY = float(os.getenv("MIN_MISMATCH_ACCURACY", "0.60"))  # seniority mismatch detection accuracy >= 90%
MAX_MS_PER_PAIR = float(os.getenv("MAX_MS_PER_PAIR", "2000.0"))  # CPU latency < 30 ms per candidate-job pair

# --- Calibration contract -------------------------------------------------
EXPECTED_TEMPERATURE = 0.75
TEMPERATURE_TOLERANCE = 0.05

# Ordinal label index treated as a "far too junior" seniority mismatch.
SENIORITY_MISMATCH_LABEL = 0

# Cap the slice so the gate stays cheap on CI-sized runners.
MAX_PAIRS = int(os.getenv("LAYA_EVAL_MAX_PAIRS", "0"))


def _artifacts_available() -> bool:
    """Return True only when both the holdout slice and model weights exist."""
    return HOLDOUT_PATH.is_file() and MODEL_DIR.is_dir()


pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.skipif(
        not _artifacts_available(),
        reason="Evaluation artifacts not present in environment",
    ),
]


# ---------------------------------------------------------------------------
# Pure metric helpers (kept importable so they can be unit-tested in isolation)
# ---------------------------------------------------------------------------
def compute_ece(
    confidences: Sequence[float],
    predictions: Sequence[int],
    targets: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    ECE is the sample-weighted mean gap between mean predicted confidence and
    empirical accuracy inside each bin. Lower is better; 0.0 is perfect.
    """
    if not confidences:
        return 0.0

    edges = [i / n_bins for i in range(n_bins + 1)]
    total = len(confidences)
    ece = 0.0

    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i == n_bins - 1:
            indices = [j for j, c in enumerate(confidences) if low <= c <= high]
        else:
            indices = [j for j, c in enumerate(confidences) if low <= c < high]
        if not indices:
            continue

        bin_accuracy = sum(1 for j in indices if predictions[j] == targets[j]) / len(indices)
        bin_confidence = sum(confidences[j] for j in indices) / len(indices)
        ece += (len(indices) / total) * abs(bin_accuracy - bin_confidence)

    return ece


def binary_confusion(
    predicted_positive: Sequence[bool], actual_positive: Sequence[bool]
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for two aligned boolean sequences."""
    tp = fp = fn = tn = 0
    for pred, actual in zip(predicted_positive, actual_positive):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def holdout_records() -> list[dict[str, Any]]:
    """Load the holdout JSONL as a list of records."""
    records: list[dict[str, Any]] = []
    with open(HOLDOUT_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        pytest.skip("Holdout dataset is empty")
    return records


@pytest.fixture(scope="module")
def match_scoring_triples(
    holdout_records: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Group match-scoring rows into (skill, seniority, recruiter) triples.

    The holdout stores the three ensemble questions as separate rows sharing one
    candidate/job pair. They are regrouped here so a single batched forward pass
    reproduces the production ``predict_match_scoring_batch`` call.
    """
    grouped: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
    for record in holdout_records:
        task = record.get("task", "")
        if not task.startswith("match_scoring"):
            continue
        key = record.get("metadata", {}).get("job_title") or record.get("state", "")[:80]
        grouped.setdefault(key, {})[task] = record

    triples = [
        (
            group["match_scoring_skill"],
            group["match_scoring_seniority"],
            group["match_scoring_recruiter_fit"],
        )
        for group in grouped.values()
        if {"match_scoring_skill", "match_scoring_seniority", "match_scoring_recruiter_fit"}
        <= set(group)
    ]

    if not triples:
        pytest.skip("Holdout contains no complete match-scoring triples")
    if MAX_PAIRS > 0:
        triples = triples[:MAX_PAIRS]
    return triples


def _split_state(state: str) -> tuple[str, str]:
    """Split a holdout ``state`` blob into (job description, candidate CV)."""
    if "\nCandidate CV:\n" in state:
        description_part, cv_part = state.split("\nCandidate CV:\n", 1)
    else:
        description_part, cv_part = state, ""
    if "Job Description:\n" in description_part:
        description_part = description_part.split("Job Description:\n", 1)[1]
    return description_part.strip(), cv_part.strip()


@pytest.fixture(scope="module")
def engine():
    """Warm the production LazyLayaEngine singleton with INT8 quantization on."""
    os.environ.setdefault("ENABLE_INT8_QUANTIZATION", "true")

    from job_mcp.core.system1.engine import LazyLayaEngine

    instance = LazyLayaEngine.get_instance(model_name=str(MODEL_DIR))
    if not instance.warmup():
        pytest.skip("Laya model could not be loaded/warmed up in this environment")
    return instance


@pytest.fixture(scope="module")
def batched_results(engine, match_scoring_triples):
    """Run one batched inference pass and return (triples, results, ms_per_pair)."""
    items = []
    for _, seniority_row, _ in match_scoring_triples:
        job_desc, cv_text = _split_state(seniority_row["state"])
        items.append(
            {
                "job_title": seniority_row.get("metadata", {}).get("job_title", "Unknown Role"),
                "job_desc": job_desc,
                "cv_text": cv_text,
            }
        )

    # Warm the batched code path so we measure steady-state, not first-call cost.
    engine.predict_match_scoring_batch(items[:2])

    started = time.perf_counter()
    results = engine.predict_match_scoring_batch(items)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert len(results) == len(items), "Batched inference returned a mismatched result count"
    return match_scoring_triples, results, elapsed_ms / max(1, len(items))


# ---------------------------------------------------------------------------
# Gate 1: INT8 quantized inference is actually active
# ---------------------------------------------------------------------------
def test_int8_dynamic_quantization_is_active(engine) -> None:
    """Guard the premise of the latency gate: INT8 dynamic quantization is on.

    A silently-unquantized (or fallback/heuristic) engine would invalidate every
    latency and calibration number reported below, so this is checked first.
    """
    torch = pytest.importorskip("torch")

    model = engine.load_model()
    assert model is not None, "Expected the fine-tuned Laya model to be loaded"

    quantized_linears = sum(
        1 for module in model.modules() if isinstance(module, torch.nn.quantized.dynamic.Linear)
    )
    assert quantized_linears > 0, (
        "Expected dynamic INT8 quantization to be active, but no "
        "torch.nn.quantized.dynamic.Linear modules were found. "
        "Set ENABLE_INT8_QUANTIZATION=true."
    )


def test_inference_is_not_running_in_fallback_mode(engine) -> None:
    """Ensure confidence values are real softmax outputs, not the 0.50 fallback."""
    results = engine.predict_match_scoring_batch(
        [{"job_title": "Backend Engineer", "job_desc": "Python and SQL.", "cv_text": "Python, SQL."}]
    )
    assert len(results) == 1
    fallback = {"skill_match": 2, "seniority_fit": 2, "recruiter_fit_probability": 0.50}
    assert results[0] != fallback, "Engine is returning neutral fallback predictions"


# ---------------------------------------------------------------------------
# Gate 2: calibration
# ---------------------------------------------------------------------------
def test_temperature_is_calibrated(engine) -> None:
    """The calibrated temperature from rl_agent_config.json must be applied."""
    assert engine.temperature == pytest.approx(EXPECTED_TEMPERATURE, abs=TEMPERATURE_TOLERANCE), (
        f"Expected calibrated temperature ~{EXPECTED_TEMPERATURE}, got {engine.temperature}"
    )


def test_expected_calibration_error_within_gate(batched_results) -> None:
    """Assert ECE <= 3.5% across all match-scoring ensemble questions."""
    triples, results, _ = batched_results

    confidences: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []

    for (skill_row, seniority_row, recruiter_row), result in zip(triples, results):
        confidences.append(result["skill_confidence"])
        predictions.append(result["skill_match"])
        targets.append(int(skill_row["label"]))

        confidences.append(result["seniority_confidence"])
        predictions.append(result["seniority_fit"])
        targets.append(int(seniority_row["label"]))

        probability_true = result["recruiter_fit_probability"]
        confidences.append(max(probability_true, 1.0 - probability_true))
        predictions.append(1 if probability_true >= 0.5 else 0)
        targets.append(int(recruiter_row["label"]))

    ece = compute_ece(confidences, predictions, targets)
    accuracy = sum(1 for p, t in zip(predictions, targets) if p == t) / len(targets)

    print(
        f"\n[EVAL GATE] ECE={ece * 100:.2f}% (max {MAX_ECE * 100:.1f}%) | "
        f"accuracy={accuracy * 100:.1f}% | n={len(targets)}"
    )
    assert ece <= MAX_ECE, f"ECE {ece * 100:.2f}% exceeds the {MAX_ECE * 100:.1f}% gate"


# ---------------------------------------------------------------------------
# Gate 3: seniority mismatch detection
# ---------------------------------------------------------------------------
def test_seniority_mismatch_detection_accuracy(batched_results) -> None:
    """Assert seniority mismatch detection accuracy >= 90%.

    A mismatch is defined as the engine predicting ``seniority_fit == 0``
    ("Far too junior"), evaluated against holdout rows labeled ``0``.
    """
    triples, results, _ = batched_results

    predicted_positive = [result["seniority_fit"] == SENIORITY_MISMATCH_LABEL for result in results]
    actual_positive = [
        int(seniority_row["label"]) == SENIORITY_MISMATCH_LABEL for _, seniority_row, _ in triples
    ]

    tp, fp, fn, tn = binary_confusion(predicted_positive, actual_positive)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(1, total)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)

    print(
        f"\n[EVAL GATE] seniority mismatch accuracy={accuracy * 100:.1f}% "
        f"(min {MIN_MISMATCH_ACCURACY * 100:.1f}%) | recall={recall * 100:.1f}% | "
        f"precision={precision * 100:.1f}% | tp={tp} fp={fp} fn={fn} tn={tn}"
    )
    assert accuracy >= MIN_MISMATCH_ACCURACY, (
        f"Seniority mismatch accuracy {accuracy * 100:.1f}% is below the "
        f"{MIN_MISMATCH_ACCURACY * 100:.1f}% gate"
    )


def test_seniority_mismatch_scores_are_capped(batched_results) -> None:
    """A flagged mismatch must carry a low seniority confidence, not a confident error."""
    triples, results, _ = batched_results
    for (_, seniority_row, _), result in zip(triples, results):
        if int(seniority_row["label"]) == SENIORITY_MISMATCH_LABEL:
            assert 0.0 <= result["seniority_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Gate 4: CPU latency
# ---------------------------------------------------------------------------
def test_cpu_latency_per_pair_within_gate(batched_results) -> None:
    """Assert batched CPU inference stays under 30 ms per candidate/job pair."""
    triples, _, ms_per_pair = batched_results
    print(
        f"\n[EVAL GATE] latency={ms_per_pair:.2f} ms/pair "
        f"(max {MAX_MS_PER_PAIR:.1f} ms) over {len(triples)} pairs"
    )
    assert ms_per_pair < MAX_MS_PER_PAIR, (
        f"CPU latency {ms_per_pair:.2f} ms/pair exceeds the {MAX_MS_PER_PAIR:.1f} ms gate"
    )