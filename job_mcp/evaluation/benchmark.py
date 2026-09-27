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

BENCHMARK_SCHEMA_VERSION = "m3-benchmark-v1"
ECE_BINS = 10
COLD_OBSERVATIONS, COLD_TIMEOUT_SECONDS = 8, 120
PUBLICATION_BUDGET_SECONDS, WARMUP_RUNS = 75 * 60, 5
WARM_ENSEMBLE_PROCESSES, WARM_ENSEMBLE_RUNS = 2, 50
BATCH_SIZES, BATCH_PROCESSES, BATCH_RUNS = (1, 4, 8, 15), 2, 10
MATCH_SCORE_PROCESSES, MATCH_SCORE_RUNS = 2, 30
FILTER_PROCESSES, FILTER_RUNS = 2, 5
REQUIRED_MATCH_SCORING_TASKS = frozenset(
    {"match_scoring_skill", "match_scoring_seniority", "match_scoring_recruiter_fit"}
)
REQUIRED_ARTIFACTS = frozenset({"holdout", "training", "model"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_DEPENDENCIES = ("job-mcp", "torch", "transformers")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


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
    conflicts = []
    for state, values in sorted(pair_states.items()):
        for task_index, task in enumerate(
            (
                "match_scoring_skill",
                "match_scoring_seniority",
                "match_scoring_recruiter_fit",
            )
        ):
            labels = {
                _canonical_json(pair[task_index].get("label")).decode()
                for pair in values
            }
            if len(labels) > 1:
                conflicts.append(
                    {
                        "task": task,
                        "state_sha256": _state_digest(state),
                        "labels": sorted(labels),
                    }
                )
    # Training often lacks complete triples; still identify task/state conflicts deterministically.
    by_task_state: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in [*training, *holdout]:
        by_task_state[(str(row.get("task")), str(row.get("state")))].add(
            _canonical_json(row.get("label")).decode()
        )
    for (task, state), labels in sorted(by_task_state.items()):
        entry = {
            "task": task,
            "state_sha256": _state_digest(state),
            "labels": sorted(labels),
        }
        if len(labels) > 1 and entry not in conflicts:
            conflicts.append(entry)
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


def protocol_observation_counts() -> dict[str, Any]:
    return {
        "cold_load": COLD_OBSERVATIONS,
        "first_inference": COLD_OBSERVATIONS,
        "warm_ensemble": 100,
        "batch": {str(size): 20 for size in BATCH_SIZES},
        "calculate_match_score": 60,
        "filter_jobs": 10,
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


def execute_protocol(
    worker_command: Sequence[str], deadline_ns: int | None = None
) -> dict[str, Any]:
    """Run every fixed boundary in isolated subprocesses under a monotonic deadline."""
    deadline_ns = (
        deadline_ns or time.monotonic_ns() + PUBLICATION_BUDGET_SECONDS * 1_000_000_000
    )
    raw: list[dict[str, Any]] = []
    origins: set[str] = set()
    devices: set[str] = set()
    plans = [("cold_load", "cold_load_first_inference", 1, COLD_OBSERVATIONS, True)]
    plans += [
        (
            "warm_ensemble",
            "warm_ensemble",
            WARM_ENSEMBLE_RUNS,
            WARM_ENSEMBLE_PROCESSES,
            False,
        )
    ]
    plans += [
        (f"batch_{size}", "model_only_forward", BATCH_RUNS, BATCH_PROCESSES, False)
        for size in BATCH_SIZES
    ]
    plans += [
        (
            "calculate_match_score",
            "calculate_match_score",
            MATCH_SCORE_RUNS,
            MATCH_SCORE_PROCESSES,
            False,
        ),
        ("filter_jobs", "filter_jobs", FILTER_RUNS, FILTER_PROCESSES, False),
    ]
    for input_name, boundary, runs, processes, cold in plans:
        for process_index in range(processes):
            remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining <= 0:
                raise TimeoutError("publication hard budget exceeded")
            timeout = min(COLD_TIMEOUT_SECONDS, remaining) if cold else remaining
            request = {
                "input": input_name,
                "boundary": boundary,
                "runs": runs,
                "warmups": 0 if cold else WARMUP_RUNS,
            }
            reply = _worker_call(worker_command, request, timeout)
            devices.add(str(reply.get("device")).lower())
            samples = reply.get("samples")
            expected_count = 2 if cold else runs
            if not isinstance(samples, list) or len(samples) != expected_count:
                raise RuntimeError("benchmark worker returned invalid sample count")
            for sample in samples:
                expected_boundaries = (
                    {"cold_load", "first_inference"} if cold else {boundary}
                )
                if (
                    sample.get("boundary") not in expected_boundaries
                    or sample.get("input") != input_name
                ):
                    raise RuntimeError("benchmark worker returned invalid boundary")
                if sample.get("boundary") != "cold_load":
                    origins.update(_validated_origins([sample.get("origin")], "timed"))
                raw.append(
                    {
                        **sample,
                        "process": reply.get("pid"),
                        "process_index": process_index,
                        "warmups_excluded": 0 if cold else WARMUP_RUNS,
                    }
                )
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
        "observations": protocol_observation_counts(),
        "raw_samples": raw,
        "quality": {
            "inference_origins": sorted(origins),
            "metrics": quality_reply.get("metrics", {}),
        },
        "environment": quality_reply.get("runtime_environment", {}),
        "runtime_model": quality_reply.get("runtime_model", {}),
    }


def _valid_raw_samples(samples: Any) -> bool:
    """Verify every boundary, process partition, origin, and warmup count."""
    if not isinstance(samples, list):
        return False
    expected: dict[tuple[str, str], tuple[int, int, int, bool]] = {
        ("cold_load", "cold_load"): (8, 1, 0, False),
        ("cold_load", "first_inference"): (8, 1, 0, True),
        ("warm_ensemble", "warm_ensemble"): (2, 50, 5, True),
        ("calculate_match_score", "calculate_match_score"): (2, 30, 5, True),
        ("filter_jobs", "filter_jobs"): (2, 5, 5, True),
    }
    expected.update(
        {
            (f"batch_{size}", "model_only_forward"): (2, 10, 5, True)
            for size in BATCH_SIZES
        }
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        if not isinstance(sample, Mapping):
            return False
        grouped[(str(sample.get("input")), str(sample.get("boundary")))].append(sample)
    if set(grouped) != set(expected):
        return False
    for key, (processes, runs, warmups, needs_origin) in expected.items():
        rows = grouped[key]
        if len(rows) != processes * runs:
            return False
        process_ids = {row.get("process") for row in rows}
        if None in process_ids or len(process_ids) != processes:
            return False
        for process_index in range(processes):
            partition = [
                row for row in rows if row.get("process_index") == process_index
            ]
            if len(partition) != runs or {row.get("run") for row in partition} != set(
                range(runs)
            ):
                return False
            if any(
                row.get("warmups_excluded") != warmups
                or not isinstance(row.get("elapsed_ns"), int)
                or row["elapsed_ns"] < 0
                or (needs_origin and row.get("origin") != "real")
                or (not needs_origin and "origin" in row)
                for row in partition
            ):
                return False
    return True


def _valid_calibration(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"n_bins", "ece", "bins"}:
        return False
    if value.get("n_bins") != ECE_BINS or not isinstance(
        value.get("ece"), (int, float)
    ):
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
        if not isinstance(item.get("count"), int) or item["count"] < 0:
            return False
        if any(
            not isinstance(item.get(key), (int, float)) or not 0 <= item[key] <= 1
            for key in ("mean_confidence", "accuracy")
        ):
            return False
    return 0 <= value["ece"] <= 1


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
    if not all(isinstance(value[key], int) and value[key] >= 0 for key in counts):
        return False
    if value["positive_count"] != value["tp"] + value["fn"]:
        return False
    if value["negative_count"] != value["fp"] + value["tn"]:
        return False
    return all(
        isinstance(value[key], (int, float)) and 0 <= value[key] <= 1 for key in rates
    )


def _valid_quality(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"source", "views"}:
        return False
    source = value.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "total_record_count",
        "match_scoring_record_count",
    }:
        return False
    total_count = source.get("total_record_count")
    match_count = source.get("match_scoring_record_count")
    if (
        type(total_count) is not int
        or total_count < 0
        or type(match_count) is not int
        or match_count < 0
        or match_count > total_count
        or match_count % 3
    ):
        return False
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
            not isinstance(view.get("pair_count"), int)
            or view["pair_count"] < 0
            or not isinstance(view.get("record_count"), int)
            or view["record_count"] < 0
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
        "observations",
        "raw_samples",
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
        or type(threads.get("requested")) is not int
        or threads["requested"] <= 0
        or type(threads.get("effective")) is not int
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
        or not isinstance(publication.get("elapsed_seconds"), (int, float))
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
            or type(model.get("int8_linear_modules")) is not int
            or model["int8_linear_modules"] <= 0
        ):
            errors.append("INT8 model inspection failed")
        temperature = model.get("temperature")
        configured_temperature = model.get("temperature_config_value")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
            or isinstance(configured_temperature, bool)
            or not isinstance(configured_temperature, (int, float))
            or not math.isfinite(configured_temperature)
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
    if result.get("observations") != protocol_observation_counts():
        errors.append("invalid benchmark observation counts")
    if not _valid_raw_samples(result.get("raw_samples")):
        errors.append("invalid raw benchmark samples")
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
