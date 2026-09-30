"""Offline, schema-versioned primitives and process protocol for Milestone 3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

BENCHMARK_SCHEMA_VERSION = "m3-benchmark-v2"
ECE_BINS = 10
COLD_OBSERVATIONS, COLD_TIMEOUT_SECONDS = 8, 120
PUBLICATION_BUDGET_SECONDS, WARMUP_RUNS = 75 * 60, 5
MODEL_FORWARD_PROCESSES, MODEL_FORWARD_RUNS = 2, 50
WARM_ENSEMBLE_PROCESSES, WARM_ENSEMBLE_RUNS = 2, 50
BATCH_SIZES, BATCH_PROCESSES, BATCH_RUNS = (1, 4, 8, 15), 2, 10
INPUT_ROTATION_SIZE = 16
MATCH_SCORE_PROCESSES, MATCH_SCORE_RUNS = 2, 30
FILTER_PROCESSES, FILTER_RUNS = 2, 5
REQUIRED_MATCH_SCORING_TASKS = frozenset(
    {"match_scoring_skill", "match_scoring_seniority", "match_scoring_recruiter_fit"}
)
REQUIRED_ARTIFACTS = frozenset({"holdout", "training", "model"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_DEPENDENCIES = ("job-mcp", "torch", "transformers")
MAX_PUBLICATION_COUNT = 1_000_000_000
# Latency is a duration, not a count, so it must never inherit count bounds.
# No legitimate raw sample can outlast the whole publication run, so durations
# are capped at the benchmark's own execution limit (4.5e12 ns = 75 minutes),
# which comfortably exceeds the 120s cold subprocess timeout while still
# rejecting pathological JSON integers that would break float conversion.
MAX_LATENCY_NS = PUBLICATION_BUDGET_SECONDS * 1_000_000_000
FIXED_INPUT_CORPUS = tuple(
    {
        "job_title": f"Backend Engineer {index + 1}",
        "job_desc": f"Python service platform fixture {index + 1}",
        "cv_text": f"Backend developer with Python experience {index + 1}",
    }
    for index in range(INPUT_ROTATION_SIZE)
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def benchmark_input(index: int) -> dict[str, str]:
    if not 0 <= index < INPUT_ROTATION_SIZE:
        raise ValueError("benchmark input index is outside the fixed corpus")
    return dict(FIXED_INPUT_CORPUS[index])


def benchmark_input_digest(index: int) -> str:
    return hashlib.sha256(_canonical_json(benchmark_input(index))).hexdigest()


def calibration_summary(
    confidences: Sequence[float],
    predictions: Sequence[int],
    targets: Sequence[int],
    n_bins: int = ECE_BINS,
) -> dict[str, Any]:
    """Compute the canonical ten-bin [low, high), final-inclusive ECE."""
    if n_bins != ECE_BINS:
        raise ValueError("Milestone 3 ECE requires exactly 10 bins")
    if len(confidences) != len(predictions) or len(predictions) != len(targets):
        raise ValueError("confidence, prediction, and target lengths must match")
    if any(not 0 <= float(value) <= 1 for value in confidences):
        raise ValueError("confidence values must be within [0, 1]")
    bins, ece, total = [], 0.0, len(confidences)
    for index in range(n_bins):
        low, high = index / n_bins, (index + 1) / n_bins
        members = [
            i
            for i, value in enumerate(confidences)
            if low <= value < high or (index == n_bins - 1 and value == high)
        ]
        count = len(members)
        mean = sum(float(confidences[i]) for i in members) / count if count else 0.0
        accuracy = (
            sum(predictions[i] == targets[i] for i in members) / count if count else 0.0
        )
        bins.append(
            {
                "low": low,
                "high": high,
                "count": count,
                "mean_confidence": mean,
                "accuracy": accuracy,
            }
        )
        ece += count / total * abs(accuracy - mean) if total else 0.0
    return {"n_bins": n_bins, "ece": ece, "bins": bins}


def compute_ece(
    confidences: Sequence[float],
    predictions: Sequence[int],
    targets: Sequence[int],
    n_bins: int = ECE_BINS,
) -> float:
    return float(calibration_summary(confidences, predictions, targets, n_bins)["ece"])


def binary_metrics(
    predicted_positive: Sequence[bool], actual_positive: Sequence[bool]
) -> dict[str, float | int]:
    if len(predicted_positive) != len(actual_positive):
        raise ValueError("prediction and target lengths must match")
    tp = fp = fn = tn = 0
    for predicted, actual in zip(predicted_positive, actual_positive):
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_count": tp + fn,
        "negative_count": fp + tn,
        "accuracy": (tp + tn) / len(actual_positive) if actual_positive else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def _match_scoring_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(row) for row in records if row.get("task") in REQUIRED_MATCH_SCORING_TASKS
    ]


def build_match_scoring_triples(
    records: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    rows = _match_scoring_records(records)
    if len(rows) % 3:
        raise ValueError(
            "match-scoring records are not divisible into complete triples"
        )
    triples = []
    for offset in range(0, len(rows), 3):
        group = rows[offset : offset + 3]
        if {row.get("task") for row in group} != REQUIRED_MATCH_SCORING_TASKS:
            raise ValueError(
                f"incomplete or duplicate tasks in match-scoring triple at offset {offset}"
            )
        if any(
            not isinstance(row.get("state"), str) or not row["state"] for row in group
        ):
            raise ValueError(f"match-scoring triple at offset {offset} has no state")
        if len({_state_digest(row["state"]) for row in group}) != 1:
            raise ValueError(
                f"match-scoring triple at offset {offset} has no canonical state identity"
            )
        tasks = {row["task"]: row for row in group}
        triples.append(
            (
                tasks["match_scoring_skill"],
                tasks["match_scoring_seniority"],
                tasks["match_scoring_recruiter_fit"],
            )
        )
    return triples


def _state_digest(state: str) -> str:
    return hashlib.sha256(state.encode()).hexdigest()


def audit_holdout_overlap(
    training: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Audit overlap plus duplicate *pair* clusters, never the expected three rows."""
    training_ids = {
        (
            str(row.get("task")),
            str(row.get("state")),
            _canonical_json(row.get("label")).decode(),
        )
        for row in training
    }
    task_states = {(str(row.get("task")), str(row.get("state"))) for row in training}
    exact = [
        row
        for row in holdout
        if (
            str(row.get("task")),
            str(row.get("state")),
            _canonical_json(row.get("label")).decode(),
        )
        in training_ids
    ]
    overlap = [
        row
        for row in holdout
        if (str(row.get("task")), str(row.get("state"))) in task_states
    ]
    # Reconstruct each corpus independently: a partial training subset must not
    # shift holdout triple boundaries or masquerade as a duplicate pair.
    try:
        pairs = build_match_scoring_triples([dict(row) for row in training])
    except ValueError:
        pairs = []
    try:
        pairs += build_match_scoring_triples([dict(row) for row in holdout])
    except ValueError:
        pairs += []
    pair_states: dict[
        str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for pair in pairs:
        pair_states[pair[0]["state"]].append(pair)
    duplicates = [
        {"state_sha256": _state_digest(state), "pair_count": len(values)}
        for state, values in sorted(pair_states.items())
        if len(values) > 1
    ]
    # Canonical conflict provenance: approved match-scoring tasks only, aggregated
    # by (task, state) into one entry with a merged sorted-unique label set, so the
    # publication output can never contradict the fail-closed validator.
    conflict_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for state, values in pair_states.items():
        for task_index, task in enumerate(
            (
                "match_scoring_skill",
                "match_scoring_seniority",
                "match_scoring_recruiter_fit",
            )
        ):
            conflict_labels[(task, _state_digest(state))].update(
                _canonical_json(pair[task_index].get("label")).decode() for pair in values
            )
    # Training often lacks complete triples; still identify task/state conflicts
    # deterministically from the combined corpus.
    for row in [*training, *holdout]:
        task = str(row.get("task"))
        if task not in REQUIRED_MATCH_SCORING_TASKS:
            continue
        conflict_labels[(task, _state_digest(str(row.get("state"))))].add(
            _canonical_json(row.get("label")).decode()
        )
    conflicts = [
        {"task": task, "state_sha256": state_sha256, "labels": sorted(labels)}
        for (task, state_sha256), labels in sorted(conflict_labels.items())
        if len(labels) > 1
    ]
    return {
        "exact_overlap_count": len(exact),
        "task_state_overlap_count": len(overlap),
        "duplicate_shared_state_clusters": duplicates,
        "conflicting_labels": conflicts,
        "conflicting_label_count": len(conflicts),
        "overlap_state_sha256": sorted(
            {_state_digest(str(row.get("state"))) for row in overlap}
        ),
    }


def _view(
    records: Sequence[Mapping[str, Any]], name: str, excluded: Sequence[str]
) -> dict[str, Any]:
    match_records = _match_scoring_records(records)
    return {
        "name": name,
        "diagnostic": True,
        "independence": "diagnostic/not independent",
        "pair_count": len(build_match_scoring_triples(match_records)),
        "record_count": len(match_records),
        "excluded_clusters": len(excluded),
        "excluded_pairs": 0,
        "remaining_checksum": hashlib.sha256(
            _canonical_json(match_records)
        ).hexdigest(),
    }


def primary_and_sensitivity_views(
    holdout: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    excluded = set(audit["overlap_state_sha256"])
    retained = [
        row for row in holdout if _state_digest(str(row.get("state"))) not in excluded
    ]
    primary, sensitivity = (
        _view(holdout, "primary_reproduction", []),
        _view(retained, "sensitivity_b", sorted(excluded)),
    )
    sensitivity["excluded_pairs"] = primary["pair_count"] - sensitivity["pair_count"]
    return {"primary": primary, "sensitivity_b": sensitivity}


DELTA_METRICS = (
    "ece",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
)


def quality_delta_b_minus_a(views: Mapping[str, Any]) -> dict[str, float]:
    primary = views["primary"]
    sensitivity = views["sensitivity_b"]
    return {
        metric: (
            sensitivity["calibration"]["ece"] - primary["calibration"]["ece"]
            if metric == "ece"
            else sensitivity["seniority_mismatch"][metric]
            - primary["seniority_mismatch"][metric]
        )
        for metric in DELTA_METRICS
    }


def canonical_file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_model_tree_checksum(path: Path) -> dict[str, Any]:
    files = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": canonical_file_checksum(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return {
        "sha256": hashlib.sha256(_canonical_json(files)).hexdigest(),
        "files": files,
    }


def percentile_summary(
    samples: Sequence[int | float], unit: str = "ns"
) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples cannot be empty")
    values = sorted(float(value) for value in samples)

    def quantile(p: float) -> float:
        index = (len(values) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)

    return {
        "unit": unit,
        "count": len(values),
        "min": values[0],
        "mean": statistics.fmean(values),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": values[-1],
        "p99_label": "low-N diagnostic" if len(values) < 100 else "type-7 estimate",
    }


def protocol_observation_counts() -> dict[str, int]:
    return {
        "cold_load": COLD_OBSERVATIONS,
        "first_inference": COLD_OBSERVATIONS,
        "model_forward": MODEL_FORWARD_PROCESSES * MODEL_FORWARD_RUNS,
        "warm_ensemble": WARM_ENSEMBLE_PROCESSES * WARM_ENSEMBLE_RUNS,
        **{
            f"batch_{size}": BATCH_PROCESSES * BATCH_RUNS
            for size in BATCH_SIZES
        },
        "calculate_match_score": MATCH_SCORE_PROCESSES * MATCH_SCORE_RUNS,
        "filter_jobs": FILTER_PROCESSES * FILTER_RUNS,
    }


def dependency_versions() -> dict[str, str]:
    """Capture the installed runtime packages that determine model execution."""
    versions: dict[str, str] = {}
    for package in REQUIRED_DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def environment_capture(
    actual_device: str = "cpu",
    *,
    torch_num_threads_requested: int | None = None,
    torch_num_threads_effective: int | None = None,
) -> dict[str, Any]:
    accelerator = "cpu" if actual_device.lower() == "cpu" else "gpu"
    if torch_num_threads_requested is None:
        try:
            torch_num_threads_requested = int(os.getenv("TORCH_NUM_THREADS", "4"))
        except ValueError:
            torch_num_threads_requested = 0
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "accelerator": accelerator,
        "actual_device": actual_device.lower(),
        "dependencies": dependency_versions(),
        "torch_num_threads": {
            "requested": torch_num_threads_requested,
            "effective": torch_num_threads_effective,
        },
    }


def source_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def empty_manifest(artifacts: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A complete serialized manifest with no filesystem paths."""
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "pending",
        "source": {"revision": source_revision()},
        "environment": environment_capture(),
        "artifacts": dict(artifacts or {}),
        "model": {
            "quantization": {"requested": None, "effective": "uninspected"},
            "int8_linear_modules": 0,
            "temperature": None,
            "temperature_config_value": None,
            "temperature_source": "uninspected",
            "temperature_config_sha256": None,
        },
        "lineage": {"parent_result_sha256": None},
        "protocol": {
            "cold_observations": 8,
            "cold_timeout_seconds": 120,
            "publication_budget_seconds": 4500,
            "warmups_excluded": 5,
            "model_forward": [2, 50],
            "warm_ensemble": [2, 50],
            "batch": [2, 10, [1, 4, 8, 15]],
            "match_score": [2, 30],
            "filter_jobs": [2, 5],
        },
    }


def _has_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute()
    if isinstance(value, Mapping):
        return any(_has_absolute_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_absolute_path(item) for item in value)
    return False


def _worker_call(
    command: Sequence[str], request: Mapping[str, Any], timeout: float
) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env={**os.environ, "NO_PROXY": "*"},
    )
    if completed.returncode:
        raise RuntimeError("benchmark worker failed")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("benchmark worker emitted invalid JSON") from error
    if result.get("status") != "success":
        raise RuntimeError("benchmark worker unsuccessful")
    return result


def _validated_origins(values: Any, context: str) -> set[str]:
    if (
        not isinstance(values, list)
        or not values
        or any(value != "real" for value in values)
    ):
        raise RuntimeError(f"invalid {context} inference origin")
    return set(values)


def _lane_specs() -> dict[str, tuple[str, int, int, int, int | None]]:
    """Return boundary, run count, samples/run, warmups/run, and batch size."""
    return {
        "cold_load": ("cold_load", COLD_OBSERVATIONS, 1, 0, None),
        "first_inference": ("first_inference", COLD_OBSERVATIONS, 1, 0, None),
        "model_forward": (
            "model_only_forward",
            MODEL_FORWARD_PROCESSES,
            MODEL_FORWARD_RUNS,
            WARMUP_RUNS,
            1,
        ),
        "warm_ensemble": (
            "warm_ensemble",
            WARM_ENSEMBLE_PROCESSES,
            WARM_ENSEMBLE_RUNS,
            WARMUP_RUNS,
            None,
        ),
        **{
            f"batch_{size}": (
                "model_only_forward",
                BATCH_PROCESSES,
                BATCH_RUNS,
                WARMUP_RUNS,
                size,
            )
            for size in BATCH_SIZES
        },
        "calculate_match_score": (
            "calculate_match_score",
            MATCH_SCORE_PROCESSES,
            MATCH_SCORE_RUNS,
            WARMUP_RUNS,
            None,
        ),
        "filter_jobs": (
            "filter_jobs",
            FILTER_PROCESSES,
            FILTER_RUNS,
            WARMUP_RUNS,
            None,
        ),
    }


def _lane_metadata(identity: str, batch_size: int | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "latency_unit": "ns",
        "input_rotation": {
            "size": INPUT_ROTATION_SIZE,
            "strategy": "round_robin_wrap",
            "corpus_digests": [
                benchmark_input_digest(index) for index in range(INPUT_ROTATION_SIZE)
            ],
        },
    }
    if identity == "model_forward":
        metadata["batch_size"] = 1
    elif identity.startswith("batch_"):
        metadata.update(
            {"batch_size": batch_size, "throughput_unit": "pairs_per_second"}
        )
    return metadata


def _build_lanes(raw: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    for identity, (boundary, run_count, samples_per_run, warmups, batch_size) in (
        _lane_specs().items()
    ):
        samples = raw[identity]
        summary = percentile_summary([sample["elapsed_ns"] for sample in samples])
        lane: dict[str, Any] = {
            "identity": identity,
            "boundary": boundary,
            "protocol": {
                "run_count": run_count,
                "samples_per_run": samples_per_run,
                "warmups_excluded_per_run": warmups,
            },
            "raw_samples": samples,
            "measured_count": len(samples),
            "run_ids": [f"{identity}-run-{index}" for index in range(run_count)],
            "summary": summary,
            "percentile_resolution": {
                "method": "Hyndman-Fan type 7 linear interpolation",
                "sample_count": len(samples),
                "p99": summary["p99_label"],
            },
            "metadata": _lane_metadata(identity, batch_size),
        }
        if identity.startswith("batch_"):
            lane["throughput"] = percentile_summary(
                [sample["throughput_pairs_per_second"] for sample in samples],
                "pairs_per_second",
            )
        lanes[identity] = lane
    return lanes


def execute_protocol(
    worker_command: Sequence[str], deadline_ns: int | None = None
) -> dict[str, Any]:
    """Run every fixed boundary in isolated subprocesses under a monotonic deadline."""
    deadline_ns = (
        deadline_ns or time.monotonic_ns() + PUBLICATION_BUDGET_SECONDS * 1_000_000_000
    )
    raw: dict[str, list[dict[str, Any]]] = {
        identity: [] for identity in _lane_specs()
    }
    origins: set[str] = set()
    devices: set[str] = set()
    plans = [
        ("cold", "cold_load_first_inference", 1, COLD_OBSERVATIONS, True, None),
        (
            "model_forward",
            "model_only_forward",
            MODEL_FORWARD_RUNS,
            MODEL_FORWARD_PROCESSES,
            False,
            1,
        ),
        (
            "warm_ensemble",
            "warm_ensemble",
            WARM_ENSEMBLE_RUNS,
            WARM_ENSEMBLE_PROCESSES,
            False,
            None,
        ),
        *[
            (
                f"batch_{size}",
                "model_only_forward",
                BATCH_RUNS,
                BATCH_PROCESSES,
                False,
                size,
            )
            for size in BATCH_SIZES
        ],
        (
            "calculate_match_score",
            "calculate_match_score",
            MATCH_SCORE_RUNS,
            MATCH_SCORE_PROCESSES,
            False,
            None,
        ),
        ("filter_jobs", "filter_jobs", FILTER_RUNS, FILTER_PROCESSES, False, None),
    ]
    for identity, boundary, samples_per_run, run_count, cold, batch_size in plans:
        for run_index in range(run_count):
            remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining <= 0:
                raise TimeoutError("publication hard budget exceeded")
            timeout = min(COLD_TIMEOUT_SECONDS, remaining) if cold else remaining
            request = {
                "input": identity,
                "boundary": boundary,
                "runs": samples_per_run,
                "warmups": 0 if cold else WARMUP_RUNS,
            }
            if batch_size is not None:
                request["batch_size"] = batch_size
            request["input_payload"] = benchmark_input(
                (run_index * samples_per_run) % INPUT_ROTATION_SIZE
            )
            request["input_payloads"] = [
                benchmark_input(
                    (run_index * samples_per_run + sample_index)
                    % INPUT_ROTATION_SIZE
                )
                for sample_index in range(samples_per_run)
            ]
            reply = _worker_call(worker_command, request, timeout)
            devices.add(str(reply.get("device")).lower())
            samples = reply.get("samples")
            expected_count = 2 if cold else samples_per_run
            if not isinstance(samples, list) or len(samples) != expected_count:
                raise RuntimeError("benchmark worker returned invalid sample count")
            for worker_sample in samples:
                lane_identity = (
                    str(worker_sample.get("boundary")) if cold else identity
                )
                expected_boundary = _lane_specs().get(lane_identity, (None,))[0]
                if (
                    lane_identity not in raw
                    or worker_sample.get("boundary") != expected_boundary
                    or worker_sample.get("input") != identity
                ):
                    raise RuntimeError("benchmark worker returned invalid boundary")
                if lane_identity != "cold_load":
                    origins.update(
                        _validated_origins([worker_sample.get("origin")], "timed")
                    )
                sample_index = worker_sample.get("run")
                if (
                    type(sample_index) is not int
                    or sample_index < 0
                    or sample_index >= samples_per_run
                ):
                    raise RuntimeError("benchmark worker returned invalid sample index")
                ordinal = run_index * samples_per_run + sample_index
                input_index = ordinal % INPUT_ROTATION_SIZE
                sample = {
                    "lane": lane_identity,
                    "boundary": expected_boundary,
                    "run_id": f"{lane_identity}-run-{run_index}",
                    "sample_index": sample_index,
                    "input_id": f"pair-{input_index:02d}",
                    "input_payload_sha256": worker_sample.get(
                        "input_payload_sha256"
                    ),
                    "elapsed_ns": worker_sample.get("elapsed_ns"),
                    "warmup": False,
                    "warmups_excluded": 0 if cold else WARMUP_RUNS,
                    "process": reply.get("pid"),
                }
                if sample["input_payload_sha256"] != benchmark_input_digest(input_index):
                    raise RuntimeError("benchmark worker returned mismatched input payload")
                if lane_identity != "cold_load":
                    sample["origin"] = worker_sample.get("origin")
                if identity.startswith("batch_"):
                    elapsed_ns = sample["elapsed_ns"]
                    if not isinstance(elapsed_ns, int) or elapsed_ns <= 0:
                        raise RuntimeError("benchmark worker returned invalid elapsed time")
                    sample["batch_size"] = batch_size
                    sample["throughput_pairs_per_second"] = (
                        batch_size * 1_000_000_000 / elapsed_ns
                    )
                raw[lane_identity].append(sample)
    remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
    if remaining <= 0:
        raise TimeoutError("publication hard budget exceeded")
    quality_reply = _worker_call(
        worker_command,
        {
            "mode": "quality",
            "input": "quality",
            "boundary": "quality",
            "runs": 0,
            "warmups": 0,
        },
        remaining,
    )
    devices.add(str(quality_reply.get("device")).lower())
    origins.update(_validated_origins(quality_reply.get("origins"), "quality"))
    if devices != {"cpu"}:
        raise RuntimeError("GPU is not permitted")
    return {
        "lanes": _build_lanes(raw),
        "quality": {
            "inference_origins": sorted(origins),
            "metrics": quality_reply.get("metrics", {}),
        },
        "environment": quality_reply.get("runtime_environment", {}),
        "runtime_model": quality_reply.get("runtime_model", {}),
    }


def _valid_lanes(value: Any) -> bool:
    """Fail closed over lane identity, protocol, raw samples, and summaries."""
    specs = _lane_specs()
    if not isinstance(value, Mapping) or set(value) != set(specs):
        return False
    common_keys = {
        "identity",
        "boundary",
        "protocol",
        "raw_samples",
        "measured_count",
        "run_ids",
        "summary",
        "percentile_resolution",
        "metadata",
    }
    for identity, (boundary, run_count, samples_per_run, warmups, batch_size) in (
        specs.items()
    ):
        lane = value.get(identity)
        expected_keys = common_keys | (
            {"throughput"} if identity.startswith("batch_") else set()
        )
        if not isinstance(lane, Mapping) or set(lane) != expected_keys:
            return False
        expected_protocol = {
            "run_count": run_count,
            "samples_per_run": samples_per_run,
            "warmups_excluded_per_run": warmups,
        }
        if any(
            type(expected_protocol[key]) is not int
            or expected_protocol[key] < 0
            for key in expected_protocol
        ):
            return False
        protocol = lane.get("protocol")
        if (
            not isinstance(protocol, Mapping)
            or set(protocol) != set(expected_protocol)
            or any(
                type(protocol[key]) is not int or protocol[key] < 0
                for key in expected_protocol
            )
        ):
            return False
        expected_run_ids = [f"{identity}-run-{index}" for index in range(run_count)]
        samples = lane.get("raw_samples")
        if (
            lane.get("identity") != identity
            or lane.get("boundary") != boundary
            or lane.get("protocol") != expected_protocol
            or lane.get("measured_count") != run_count * samples_per_run
            or lane.get("run_ids") != expected_run_ids
            or lane.get("metadata") != _lane_metadata(identity, batch_size)
            or not isinstance(samples, list)
            or len(samples) != run_count * samples_per_run
        ):
            return False
        if identity.startswith("batch_") and type(
            lane["metadata"].get("batch_size")
        ) is not int:
            return False
        if any(not isinstance(sample, Mapping) for sample in samples):
            return False
        expected_sample_keys = {
            "lane",
            "boundary",
            "run_id",
            "sample_index",
            "input_id",
            "input_payload_sha256",
            "elapsed_ns",
            "warmup",
            "warmups_excluded",
            "process",
        }
        if identity != "cold_load":
            expected_sample_keys.add("origin")
        if identity.startswith("batch_"):
            expected_sample_keys.update(
                {"batch_size", "throughput_pairs_per_second"}
            )
        run_processes: list[Any] = []
        for run_index, run_id in enumerate(expected_run_ids):
            partition = [sample for sample in samples if sample.get("run_id") == run_id]
            # Process identity must be validated before it is ever hashed, so
            # unhashable hostile JSON is rejected rather than raising TypeError.
            process_values = [sample.get("process") for sample in partition]
            if any(not _valid_process_id(value) for value in process_values):
                return False
            process_ids = set(process_values)
            if len(partition) != samples_per_run or len(process_ids) != 1:
                return False
            run_processes.extend(process_ids)
            for sample_index, sample in enumerate(partition):
                ordinal = run_index * samples_per_run + sample_index
                input_index = ordinal % INPUT_ROTATION_SIZE
                if set(sample) != expected_sample_keys:
                    return False
                if (
                    sample.get("lane") != identity
                    or sample.get("boundary") != boundary
                    or type(sample.get("sample_index")) is not int
                    or sample.get("sample_index") != sample_index
                    or sample.get("input_id")
                    != f"pair-{input_index:02d}"
                    or sample.get("input_payload_sha256")
                    != benchmark_input_digest(input_index)
                    or sample.get("warmup") is not False
                    or sample.get("warmups_excluded") != warmups
                    or not _latency_ns(sample.get("elapsed_ns"))
                    or not _non_negative_int(sample.get("process"))
                    or sample.get("process") <= 0
                    or (identity != "cold_load" and sample.get("origin") != "real")
                ):
                    return False
                throughput = sample.get("throughput_pairs_per_second")
                if identity.startswith("batch_") and (
                    type(sample.get("batch_size")) is not int
                    or sample.get("batch_size") != batch_size
                    or not _finite_number(throughput)
                    or not math.isclose(
                        throughput,
                        batch_size * 1_000_000_000 / sample["elapsed_ns"],
                        rel_tol=1e-12,
                    )
                ):
                    return False
        if len(set(run_processes)) != run_count:
            return False
        expected_summary = percentile_summary(
            [sample["elapsed_ns"] for sample in samples]
        )
        expected_resolution = {
            "method": "Hyndman-Fan type 7 linear interpolation",
            "sample_count": len(samples),
            "p99": expected_summary["p99_label"],
        }
        if (
            lane.get("summary") != expected_summary
            or lane.get("percentile_resolution") != expected_resolution
        ):
            return False
        if identity.startswith("batch_") and lane.get("throughput") != percentile_summary(
            [sample["throughput_pairs_per_second"] for sample in samples],
            "pairs_per_second",
        ):
            return False
    return True


def _valid_calibration(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"n_bins", "ece", "bins"}:
        return False
    if value.get("n_bins") != ECE_BINS or not _finite_number(value.get("ece")):
        return False
    bins = value.get("bins")
    if not isinstance(bins, list) or len(bins) != ECE_BINS:
        return False
    bin_keys = {"low", "high", "count", "mean_confidence", "accuracy"}
    for index, item in enumerate(bins):
        if not isinstance(item, Mapping) or set(item) != bin_keys:
            return False
        if (
            item.get("low") != index / ECE_BINS
            or item.get("high") != (index + 1) / ECE_BINS
        ):
            return False
        if not _non_negative_int(item.get("count")):
            return False
        if any(
            not _finite_number(item.get(key)) or not 0 <= item[key] <= 1
            for key in ("mean_confidence", "accuracy")
        ):
            return False
    return 0 <= value["ece"] <= 1


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _non_negative_int(value: Any) -> bool:
    return type(value) is int and 0 <= value <= MAX_PUBLICATION_COUNT


def _latency_ns(value: Any) -> bool:
    """Strict positive duration bound, deliberately separate from count bounds."""
    return type(value) is int and 0 < value <= MAX_LATENCY_NS


def _valid_process_id(value: Any) -> bool:
    """Hashable, bounded, strictly positive process identifier."""
    return _non_negative_int(value) and value > 0


def _valid_binary_metrics(value: Any) -> bool:
    required = {
        "tp",
        "fp",
        "fn",
        "tn",
        "positive_count",
        "negative_count",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        return False
    counts = {"tp", "fp", "fn", "tn", "positive_count", "negative_count"}
    rates = required - counts
    if not all(_non_negative_int(value[key]) for key in counts):
        return False
    if value["positive_count"] != value["tp"] + value["fn"]:
        return False
    if value["negative_count"] != value["fp"] + value["tn"]:
        return False
    return all(
        _finite_number(value[key]) and 0 <= value[key] <= 1 for key in rates
    )


def _valid_source_counts(source: Any) -> bool:
    """Canonical, non-raising validation of quality source counts."""
    if not isinstance(source, Mapping) or set(source) != {
        "total_record_count",
        "match_scoring_record_count",
    }:
        return False
    total_count = source.get("total_record_count")
    match_count = source.get("match_scoring_record_count")
    return (
        _non_negative_int(total_count)
        and _non_negative_int(match_count)
        and match_count <= total_count
        and not match_count % 3
    )


def _valid_quality(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"source", "views", "delta_b_minus_a"}:
        return False
    source = value.get("source")
    if not _valid_source_counts(source):
        return False
    match_count = source["match_scoring_record_count"]
    views = value.get("views")
    if not isinstance(views, Mapping) or set(views) != {"primary", "sensitivity_b"}:
        return False
    expected = {
        "primary": ("primary_reproduction", "A-primary"),
        "sensitivity_b": ("overlap_excluded_sensitivity", "B-sensitivity"),
    }
    required = {
        "name",
        "hierarchy",
        "pair_count",
        "record_count",
        "calibration",
        "seniority_mismatch",
    }
    for key, (name, hierarchy) in expected.items():
        view = views.get(key)
        if not isinstance(view, Mapping) or set(view) != required:
            return False
        if view.get("name") != name or view.get("hierarchy") != hierarchy:
            return False
        if (
            not _non_negative_int(view.get("pair_count"))
            or not _non_negative_int(view.get("record_count"))
        ):
            return False
        if view["record_count"] != view["pair_count"] * 3:
            return False
        if key == "primary" and view["record_count"] != match_count:
            return False
        if not _valid_calibration(view.get("calibration")) or not _valid_binary_metrics(
            view.get("seniority_mismatch")
        ):
            return False
        if (
            sum(item["count"] for item in view["calibration"]["bins"])
            != view["record_count"]
        ):
            return False
        mismatch = view["seniority_mismatch"]
        if (
            mismatch["positive_count"] + mismatch["negative_count"]
            != view["pair_count"]
        ):
            return False
    delta = value.get("delta_b_minus_a")
    if not isinstance(delta, Mapping) or set(delta) != set(DELTA_METRICS):
        return False
    if any(not _finite_number(delta.get(metric)) for metric in DELTA_METRICS):
        return False
    expected_delta = quality_delta_b_minus_a(views)
    return all(
        math.isclose(
            delta[metric], expected_delta[metric], rel_tol=1e-12, abs_tol=1e-12
        )
        for metric in DELTA_METRICS
    )


def _valid_audit_views(
    audit: Any, views: Any, source: Any
) -> bool:
    """Self-contained audit/view provenance check; never raises on malformed JSON."""
    if not _valid_source_counts(source):
        return False
    total_count = source["total_record_count"]
    match_count = source["match_scoring_record_count"]
    audit_keys = {
        "exact_overlap_count",
        "task_state_overlap_count",
        "duplicate_shared_state_clusters",
        "conflicting_labels",
        "conflicting_label_count",
        "overlap_state_sha256",
    }
    if not isinstance(audit, Mapping) or set(audit) != audit_keys:
        return False
    count_keys = {
        "exact_overlap_count",
        "task_state_overlap_count",
        "conflicting_label_count",
    }
    if not all(_non_negative_int(audit.get(key)) for key in count_keys):
        return False
    digests = audit.get("overlap_state_sha256")
    if not isinstance(digests, list) or any(
        not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
        for value in digests
    ) or len(set(digests)) != len(digests):
        return False
    duplicates = audit.get("duplicate_shared_state_clusters")
    if not isinstance(duplicates, list):
        return False
    duplicate_digests: set[str] = set()
    for item in duplicates:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"state_sha256", "pair_count"}
            or not isinstance(item.get("state_sha256"), str)
            or not SHA256_PATTERN.fullmatch(item["state_sha256"])
            or not _non_negative_int(item.get("pair_count"))
            or item["pair_count"] < 2
            or item["state_sha256"] in duplicate_digests
        ):
            return False
        duplicate_digests.add(item["state_sha256"])
    conflicts = audit.get("conflicting_labels")
    if not isinstance(conflicts, list) or audit["conflicting_label_count"] != len(conflicts):
        return False
    conflict_keys: set[tuple[str, str]] = set()
    for item in conflicts:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"task", "state_sha256", "labels"}
            or not isinstance(item.get("task"), str)
            or item["task"] not in REQUIRED_MATCH_SCORING_TASKS
            or not isinstance(item.get("state_sha256"), str)
            or not SHA256_PATTERN.fullmatch(item["state_sha256"])
            or not isinstance(item.get("labels"), list)
            or len(item["labels"]) < 2
            or any(not isinstance(label, str) for label in item["labels"])
            or (item["task"], item["state_sha256"]) in conflict_keys
        ):
            return False
        conflict_keys.add((item["task"], item["state_sha256"]))
    expected_views = {"primary", "sensitivity_b"}
    if not isinstance(views, Mapping) or set(views) != expected_views:
        return False
    required_view_keys = {
        "name",
        "diagnostic",
        "independence",
        "pair_count",
        "record_count",
        "excluded_clusters",
        "excluded_pairs",
        "remaining_checksum",
    }
    for key, name in (("primary", "primary_reproduction"), ("sensitivity_b", "sensitivity_b")):
        view = views.get(key)
        if (
            not isinstance(view, Mapping)
            or set(view) != required_view_keys
            or view.get("name") != name
            or view.get("diagnostic") is not True
            or view.get("independence") != "diagnostic/not independent"
            or not _non_negative_int(view.get("pair_count"))
            or not _non_negative_int(view.get("record_count"))
            or not _non_negative_int(view.get("excluded_clusters"))
            or not _non_negative_int(view.get("excluded_pairs"))
            or not isinstance(view.get("remaining_checksum"), str)
            or not SHA256_PATTERN.fullmatch(view["remaining_checksum"])
            or view["record_count"] != view["pair_count"] * 3
        ):
            return False
    primary = views["primary"]
    sensitivity = views["sensitivity_b"]
    return not (
        audit["exact_overlap_count"] > total_count
        # Overlap counts holdout rows across every task, so its denominator must
        # be the all-task record count, never the match-scoring-only count.
        or audit["task_state_overlap_count"] > total_count
        or audit["exact_overlap_count"] > audit["task_state_overlap_count"]
        or audit["task_state_overlap_count"] < len(digests)
        or primary["pair_count"] * 3 != match_count
        or primary["record_count"] != match_count
        or primary["excluded_clusters"] != 0
        or primary["excluded_pairs"] != 0
        or sensitivity["excluded_clusters"] != len(digests)
        or sensitivity["excluded_pairs"] != primary["pair_count"] - sensitivity["pair_count"]
        or sensitivity["record_count"]
        != primary["record_count"] - sensitivity["excluded_pairs"] * 3
    )


def _quality_views_match_provenance(
    quality_metrics: Any, provenance_views: Any
) -> bool:
    if not isinstance(quality_metrics, Mapping) or not isinstance(
        provenance_views, Mapping
    ):
        return False
    metric_views = quality_metrics.get("views")
    if not isinstance(metric_views, Mapping):
        return False
    for key in ("primary", "sensitivity_b"):
        metric_view = metric_views.get(key)
        provenance_view = provenance_views.get(key)
        if not isinstance(metric_view, Mapping) or not isinstance(
            provenance_view, Mapping
        ):
            return False
        if (
            metric_view.get("pair_count") != provenance_view.get("pair_count")
            or metric_view.get("record_count") != provenance_view.get("record_count")
        ):
            return False
    return True


def validate_publication(
    result: Mapping[str, Any],
    expected_artifacts: Mapping[str, str],
    git_dirty: bool = False,
    *,
    expected_config_checksum: str | None = None,
    expected_source_counts: Mapping[str, int] | None = None,
) -> list[str]:
    """Fail closed on incomplete provenance, runtime configuration, or metrics."""
    errors: list[str] = []
    if git_dirty:
        errors.append("dirty git tree")
    required_keys = {
        "schema_version",
        "status",
        "source",
        "environment",
        "artifacts",
        "model",
        "lineage",
        "protocol",
        "publication",
        "quality",
        "lanes",
        "audit",
        "views",
    }
    for key in sorted(required_keys - set(result)):
        errors.append(f"missing required manifest key: {key}")
    if result.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        errors.append("invalid schema version")
    if result.get("status") != "success":
        errors.append("unsuccessful benchmark status")
    source = result.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"revision"}
        or not REVISION_PATTERN.fullmatch(str(source.get("revision", "")))
    ):
        errors.append("invalid source revision")
    environment = result.get("environment")
    environment_keys = {
        "python",
        "platform",
        "implementation",
        "accelerator",
        "actual_device",
        "dependencies",
        "torch_num_threads",
    }
    if not isinstance(environment, Mapping) or set(environment) != environment_keys:
        errors.append("invalid environment schema")
    if (
        not isinstance(environment, Mapping)
        or environment.get("accelerator") != "cpu"
        or environment.get("actual_device") != "cpu"
    ):
        errors.append("GPU is not permitted")
    elif any(
        not isinstance(environment.get(key), str) or not environment[key]
        for key in ("python", "platform", "implementation")
    ):
        errors.append("invalid runtime provenance")
    threads = (
        environment.get("torch_num_threads")
        if isinstance(environment, Mapping)
        else None
    )
    if (
        not isinstance(threads, Mapping)
        or set(threads) != {"requested", "effective"}
        or not _non_negative_int(threads.get("requested"))
        or threads["requested"] <= 0
        or not _non_negative_int(threads.get("effective"))
        or threads["effective"] <= 0
        or threads["effective"] != threads["requested"]
    ):
        errors.append("invalid torch thread configuration")
    dependencies = (
        environment.get("dependencies") if isinstance(environment, Mapping) else None
    )
    if (
        not isinstance(dependencies, Mapping)
        or set(dependencies) != set(REQUIRED_DEPENDENCIES)
        or any(
            not isinstance(value, str) or not value or value == "unavailable"
            for value in dependencies.values()
        )
    ):
        errors.append("invalid dependency provenance")
    if result.get("protocol") != empty_manifest({})["protocol"]:
        errors.append("invalid benchmark configuration")
    publication = result.get("publication")
    if (
        not isinstance(publication, Mapping)
        or set(publication) != {"elapsed_seconds"}
        or not _finite_number(publication.get("elapsed_seconds"))
        or not 0 <= publication["elapsed_seconds"] <= PUBLICATION_BUDGET_SECONDS
    ):
        errors.append("publication hard budget exceeded")
    quality = result.get("quality")
    if not isinstance(quality, Mapping) or quality.get("inference_origins") != ["real"]:
        errors.append("invalid quality inference origin")
    if not isinstance(quality, Mapping) or not _valid_quality(quality.get("metrics")):
        errors.append("invalid quality metrics schema")
    elif expected_source_counts is not None and quality["metrics"].get(
        "source"
    ) != dict(expected_source_counts):
        errors.append("quality source count mismatch")
    quality_metrics = quality.get("metrics") if isinstance(quality, Mapping) else None
    quality_source = quality_metrics.get("source") if isinstance(quality_metrics, Mapping) else None
    if not isinstance(quality_source, Mapping) or not _valid_audit_views(
        result.get("audit"), result.get("views"), quality_source
    ):
        errors.append("invalid quality audit/view provenance")
    elif not _quality_views_match_provenance(quality_metrics, result["views"]):
        errors.append("quality metrics/provenance view mismatch")
    model = result.get("model")
    expected_model_keys = {
        "quantization",
        "int8_linear_modules",
        "temperature",
        "temperature_config_value",
        "temperature_source",
        "temperature_config_sha256",
    }
    if not isinstance(model, Mapping) or set(model) != expected_model_keys:
        errors.append("invalid model configuration schema")
    else:
        quantization = model.get("quantization")
        if (
            not isinstance(quantization, Mapping)
            or set(quantization) != {"requested", "effective"}
            or quantization.get("requested") is not True
            or quantization.get("effective") != "dynamic-int8"
            or not _non_negative_int(model.get("int8_linear_modules"))
            or model["int8_linear_modules"] <= 0
        ):
            errors.append("INT8 model inspection failed")
        temperature = model.get("temperature")
        configured_temperature = model.get("temperature_config_value")
        if (
            not _finite_number(temperature)
            or temperature <= 0
            or not _finite_number(configured_temperature)
            or configured_temperature <= 0
            or temperature != configured_temperature
            or model.get("temperature_source")
            != "rl_agent_config.json:calibrated_temperature"
            or not SHA256_PATTERN.fullmatch(
                str(model.get("temperature_config_sha256", ""))
            )
        ):
            errors.append("invalid model calibration configuration")
        elif (
            expected_config_checksum is not None
            and model["temperature_config_sha256"] != expected_config_checksum
        ):
            errors.append("calibration config checksum mismatch")
    lineage = result.get("lineage")
    parent = (
        lineage.get("parent_result_sha256")
        if isinstance(lineage, Mapping)
        else "invalid"
    )
    if (
        not isinstance(lineage, Mapping)
        or set(lineage) != {"parent_result_sha256"}
        or (parent is not None and not SHA256_PATTERN.fullmatch(str(parent)))
    ):
        errors.append("invalid lineage schema")
    if not _valid_lanes(result.get("lanes")):
        errors.append("invalid benchmark lanes")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != REQUIRED_ARTIFACTS:
        errors.append("invalid artifact schema")
        artifacts = {}
    for name in sorted(REQUIRED_ARTIFACTS):
        value = artifacts.get(name)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            errors.append(f"invalid artifact checksum: {name}")
        if expected_artifacts.get(name) != value:
            errors.append(f"artifact checksum mismatch: {name}")
    if _has_absolute_path(result):
        errors.append("absolute filesystem path in serialized result")
    return errors
