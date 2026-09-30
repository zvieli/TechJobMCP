"""Focused, artifact-free checks for the Milestone 3 benchmark harness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from job_mcp.evaluation import benchmark
from job_mcp.evaluation.benchmark import (
    COLD_TIMEOUT_SECONDS,
    PUBLICATION_BUDGET_SECONDS,
    REQUIRED_MATCH_SCORING_TASKS,
    audit_holdout_overlap,
    binary_metrics,
    build_match_scoring_triples,
    calibration_summary,
    canonical_file_checksum,
    canonical_model_tree_checksum,
    empty_manifest,
    execute_protocol,
    percentile_summary,
    primary_and_sensitivity_views,
    quality_delta_b_minus_a,
    validate_publication,
)


def _load_runner() -> Any:
    path = Path(__file__).parents[2] / ".scripts" / "benchmark_milestone3.py"
    spec = importlib.util.spec_from_file_location("benchmark_milestone3", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(state: str, labels: tuple[int, int, int] = (4, 2, 1)) -> list[dict[str, Any]]:
    tasks = (
        "match_scoring_skill",
        "match_scoring_seniority",
        "match_scoring_recruiter_fit",
    )
    text = f"Job Description:\n{state}\nCandidate CV:\nPython developer"
    return [
        {
            "task": task,
            "state": text,
            "label": label,
            "metadata": {"job_title": "Engineer"},
        }
        for task, label in zip(tasks, labels)
    ]


def _quality_views() -> dict[str, Any]:
    calibration = calibration_summary([0.9, 0.8, 0.7], [4, 2, 1], [4, 2, 1])
    mismatch = binary_metrics([False], [False])
    return {
        "primary": {
            "name": "primary_reproduction",
            "hierarchy": "A-primary",
            "pair_count": 1,
            "record_count": 3,
            "calibration": calibration,
            "seniority_mismatch": mismatch,
        },
        "sensitivity_b": {
            "name": "overlap_excluded_sensitivity",
            "hierarchy": "B-sensitivity",
            "pair_count": 1,
            "record_count": 3,
            "calibration": calibration,
            "seniority_mismatch": mismatch,
        },
    }


def _fake_reply(request: dict[str, Any], pid: int = 1) -> dict[str, Any]:
    if request.get("mode") == "quality":
        return {
            "status": "success",
            "device": "cpu",
            "pid": pid,
            "samples": [],
            "origins": ["real"],
            "metrics": {
                "source": {
                    "total_record_count": 3,
                    "match_scoring_record_count": 3,
                },
                "views": _quality_views(),
                "delta_b_minus_a": {
                    "ece": 0.0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "specificity": 0.0,
                    "balanced_accuracy": 0.0,
                },
            },
            "runtime_model": {
                "quantization": {
                    "requested": True,
                    "effective": "dynamic-int8",
                },
                "int8_linear_modules": 2,
                "temperature": 0.75,
                "temperature_config_value": 0.75,
                "temperature_source": "rl_agent_config.json:calibrated_temperature",
                "temperature_config_sha256": "c" * 64,
            },
            "runtime_environment": {
                "python": "3.12.0",
                "platform": "test-platform",
                "implementation": "CPython",
                "accelerator": "cpu",
                "actual_device": "cpu",
                "dependencies": {
                    "job-mcp": "0.1.0",
                    "torch": "2.0.0",
                    "transformers": "4.48.0",
                },
                "torch_num_threads": {"requested": 4, "effective": 4},
            },
        }
    if request["boundary"] == "cold_load_first_inference":
        samples = [
            {
                "run": 0,
                "input": request["input"],
                "boundary": "cold_load",
                "elapsed_ns": 1,
                "input_payload_sha256": hashlib.sha256(
                    benchmark._canonical_json(
                        request.get("input_payloads", [benchmark.benchmark_input(0)])[0]
                    )
                ).hexdigest(),
            },
            {
                "run": 0,
                "input": request["input"],
                "boundary": "first_inference",
                "elapsed_ns": 2,
                "origin": "real",
                "input_payload_sha256": hashlib.sha256(
                    benchmark._canonical_json(
                        request.get("input_payloads", [benchmark.benchmark_input(0)])[0]
                    )
                ).hexdigest(),
            },
        ]
    else:
        samples = [
            {
                "run": run,
                "input": request["input"],
                "boundary": request["boundary"],
                "elapsed_ns": 1,
                "origin": "real",
                "input_payload_sha256": hashlib.sha256(
                    benchmark._canonical_json(
                        request.get("input_payloads", [benchmark.benchmark_input(0)])[run]
                    )
                ).hexdigest(),
            }
            for run in range(request["runs"])
        ]
    return {
        "status": "success",
        "device": "cpu",
        "pid": pid,
        "samples": samples,
        "origins": ["real"],
    }


def _valid_result(
    monkeypatch: pytest.MonkeyPatch,
    latency_ns: int | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    calls = 0

    def fake_call(
        command: list[str], request: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        reply = _fake_reply(request, calls)
        if latency_ns is not None:
            for sample in reply.get("samples", []):
                sample["elapsed_ns"] = latency_ns
        return reply

    monkeypatch.setattr(benchmark, "_worker_call", fake_call)
    protocol = execute_protocol(["worker"], deadline_ns=10**30)
    digest = "a" * 64
    artifacts = {"holdout": digest, "training": digest, "model": digest}
    manifest = empty_manifest(artifacts)
    manifest["source"] = {"revision": "b" * 40}
    manifest["environment"]["dependencies"] = {
        "job-mcp": "0.1.0",
        "torch": "2.0.0",
        "transformers": "4.48.0",
    }
    manifest["model"] = protocol.pop("runtime_model")
    result = {
        **manifest,
        **protocol,
        "status": "success",
        "publication": {"elapsed_seconds": 1.0},
        "audit": {
            "exact_overlap_count": 0,
            "task_state_overlap_count": 0,
            "duplicate_shared_state_clusters": [],
            "conflicting_labels": [],
            "conflicting_label_count": 0,
            "overlap_state_sha256": [],
        },
        "views": {
            "primary": {
                "name": "primary_reproduction",
                "diagnostic": True,
                "independence": "diagnostic/not independent",
                "pair_count": 1,
                "record_count": 3,
                "excluded_clusters": 0,
                "excluded_pairs": 0,
                "remaining_checksum": "d" * 64,
            },
            "sensitivity_b": {
                "name": "sensitivity_b",
                "diagnostic": True,
                "independence": "diagnostic/not independent",
                "pair_count": 1,
                "record_count": 3,
                "excluded_clusters": 0,
                "excluded_pairs": 0,
                "remaining_checksum": "e" * 64,
            },
        },
    }
    return result, artifacts


def test_calibration_uses_ten_equal_width_bins_with_final_bin_inclusive() -> None:
    result = calibration_summary([0.0, 0.1, 1.0], [0, 0, 1], [1, 1, 1])
    assert result["n_bins"] == 10
    assert result["bins"][0] == {
        "low": 0.0,
        "high": 0.1,
        "count": 1,
        "mean_confidence": 0.0,
        "accuracy": 0.0,
    }
    assert result["bins"][1]["count"] == 1
    assert result["bins"][-1]["count"] == 1
    assert result["ece"] == pytest.approx((1 / 3) * 0.1)


def test_binary_metrics_include_specificity_and_balanced_accuracy_at_zero_denominators() -> (
    None
):
    assert binary_metrics([False], [False]) == {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 1,
        "positive_count": 0,
        "negative_count": 1,
        "accuracy": 1.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "specificity": 1.0,
        "balanced_accuracy": 0.5,
    }


def test_duplicate_audit_counts_repeated_pairs_not_three_rows() -> None:
    pair = _rows("state-a")
    audit = audit_holdout_overlap([], pair)
    duplicated = audit_holdout_overlap([], pair + pair)
    assert audit["duplicate_shared_state_clusters"] == []
    state = pair[0]["state"]
    assert duplicated["duplicate_shared_state_clusters"] == [
        {"state_sha256": hashlib.sha256(state.encode()).hexdigest(), "pair_count": 2}
    ]


def test_ordered_triples_and_overlap_sensitivity_keep_primary_pairs() -> None:
    holdout = _rows("state-a") + _rows("state-b")
    audit = audit_holdout_overlap(
        [{"task": "match_scoring_skill", "state": holdout[0]["state"], "label": 99}],
        holdout,
    )
    views = primary_and_sensitivity_views(holdout, audit)
    assert len(build_match_scoring_triples(holdout)) == 2
    assert audit["task_state_overlap_count"] == 1
    assert views["primary"]["pair_count"] == 2
    assert views["sensitivity_b"]["pair_count"] == 1


def test_match_scoring_triple_requires_one_canonical_state_identity() -> None:
    malformed = _rows("state-a")
    malformed[1] = {**malformed[1], "state": _rows("state-b")[1]["state"]}
    with pytest.raises(ValueError, match="canonical state"):
        build_match_scoring_triples(malformed)


def test_checksums_are_canonical_and_never_serialize_absolute_paths(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("contents", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    assert canonical_file_checksum(artifact) == hashlib.sha256(b"contents").hexdigest()
    tree = canonical_model_tree_checksum(model)
    assert str(tmp_path) not in str(tree)
    assert tree["files"] == [
        {"path": "config.json", "sha256": hashlib.sha256(b"{}").hexdigest()}
    ]


def test_percentiles_are_type7_and_low_sample_p99_is_labeled() -> None:
    summary = percentile_summary([1, 2, 3, 4], "ns")
    assert summary["p50"] == 2.5
    assert summary["p99"] == pytest.approx(3.97)
    assert summary["p99_label"] == "low-N diagnostic"


def test_protocol_materializes_exact_lanes_with_counts_rotation_and_batch_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_call(
        command: list[str], request: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _fake_reply(request, calls)

    monkeypatch.setattr(benchmark, "_worker_call", fake_call)
    result = execute_protocol(["worker"], deadline_ns=10**30)
    expected_counts = {
        "cold_load": 8,
        "first_inference": 8,
        "model_forward": 100,
        "warm_ensemble": 100,
        "batch_1": 20,
        "batch_4": 20,
        "batch_8": 20,
        "batch_15": 20,
        "calculate_match_score": 60,
        "filter_jobs": 10,
    }
    assert set(result["lanes"]) == set(expected_counts)
    for identity, measured_count in expected_counts.items():
        lane = result["lanes"][identity]
        assert lane["identity"] == identity
        assert lane["measured_count"] == measured_count
        assert len(lane["raw_samples"]) == measured_count
        assert lane["summary"]["count"] == measured_count
        assert lane["percentile_resolution"]["sample_count"] == measured_count
        assert lane["run_ids"] == list(dict.fromkeys(
            sample["run_id"] for sample in lane["raw_samples"]
        ))
        assert all(sample["warmup"] is False for sample in lane["raw_samples"])
        assert all("input_id" in sample for sample in lane["raw_samples"])
        assert lane["metadata"]

    assert result["lanes"]["model_forward"]["protocol"] == {
        "run_count": 2,
        "samples_per_run": 50,
        "warmups_excluded_per_run": 5,
    }
    assert result["lanes"]["warm_ensemble"]["protocol"] == {
        "run_count": 2,
        "samples_per_run": 50,
        "warmups_excluded_per_run": 5,
    }
    model_samples = result["lanes"]["model_forward"]["raw_samples"]
    first_run = [
        sample for sample in model_samples if sample["run_id"] == "model_forward-run-0"
    ]
    assert [sample["input_id"] for sample in first_run[:17]] == [
        *(f"pair-{index:02d}" for index in range(16)),
        "pair-00",
    ]
    for size in (1, 4, 8, 15):
        lane = result["lanes"][f"batch_{size}"]
        assert lane["metadata"]["batch_size"] == size
        assert lane["metadata"]["throughput_unit"] == "pairs_per_second"
        assert lane["throughput"]["count"] == 20
        assert all(sample["batch_size"] == size for sample in lane["raw_samples"])
        assert all(
            sample["throughput_pairs_per_second"]
            == pytest.approx(size * 1_000_000_000 / sample["elapsed_ns"])
            for sample in lane["raw_samples"]
        )


def test_protocol_uses_cold_timeout_only_for_cold_and_validates_each_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[dict[str, Any], float]] = []

    def fake_call(
        command: list[str], request: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        calls.append((request, timeout))
        return _fake_reply(request, len(calls))

    monkeypatch.setattr(benchmark, "_worker_call", fake_call)
    time_ns = benchmark.time.monotonic_ns() + 1000 * 1_000_000_000
    result = execute_protocol(["worker"], deadline_ns=time_ns)
    assert {
        name: lane["measured_count"] for name, lane in result["lanes"].items()
    } == {
        "cold_load": 8,
        "first_inference": 8,
        "model_forward": 100,
        "warm_ensemble": 100,
        "batch_1": 20,
        "batch_4": 20,
        "batch_8": 20,
        "batch_15": 20,
        "calculate_match_score": 60,
        "filter_jobs": 10,
    }
    assert all(
        timeout <= COLD_TIMEOUT_SECONDS
        for request, timeout in calls
        if request["boundary"] == "cold_load_first_inference"
    )
    assert all(
        timeout > COLD_TIMEOUT_SECONDS
        for request, timeout in calls
        if request["boundary"] != "cold_load_first_inference"
    )
    assert result["lanes"]["cold_load"]["boundary"] == "cold_load"
    assert result["lanes"]["first_inference"]["boundary"] == "first_inference"
    assert time_ns > 0

    def fallback_call(
        command: list[str], request: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        reply = _fake_reply(request)
        if request.get("boundary") == "warm_ensemble":
            reply["samples"][0]["origin"] = "fallback"
        return reply

    monkeypatch.setattr(benchmark, "_worker_call", fallback_call)
    with pytest.raises(RuntimeError, match="origin"):
        execute_protocol(["worker"], deadline_ns=10**30)


def test_protocol_fails_immediately_on_worker_payload_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_call(
        command: list[str], request: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        reply = _fake_reply(request, calls)
        if request.get("mode") != "quality":
            reply["samples"][0]["input_payload_sha256"] = "0" * 64
        return reply

    monkeypatch.setattr(benchmark, "_worker_call", fake_call)
    with pytest.raises(RuntimeError, match="mismatched input payload"):
        execute_protocol(["worker"], deadline_ns=10**30)


def test_worker_reuses_one_engine_for_exactly_five_warmups_and_pretokenizes_model_forward() -> (
    None
):
    runner = _load_runner()

    class Tokenizer:
        calls = 0

        def __call__(self, *args: Any, **kwargs: Any) -> dict[str, str]:
            self.calls += 1
            return {"input_ids": "encoded"}

    class Model:
        calls = 0

        def __init__(self) -> None:
            self.pre_hooks: list[Any] = []
            self.post_hooks: list[Any] = []

        def register_forward_pre_hook(self, hook: Any) -> None:
            self.pre_hooks.append(hook)

        def register_forward_hook(self, hook: Any) -> None:
            self.post_hooks.append(hook)

        def __call__(self, **kwargs: Any) -> Any:
            for hook in self.pre_hooks:
                hook(self, ())
            self.calls += 1
            output = SimpleNamespace(logits=[1])
            for hook in self.post_hooks:
                hook(self, (), output)
            return output

        def parameters(self):
            yield SimpleNamespace(device="cpu")

    class Engine:
        def __init__(self) -> None:
            self.original_tokenizer = Tokenizer()
            self._tokenizer = self.original_tokenizer
            self.model = Model()

        def load_model(self) -> Model:
            return self.model

        def predict_match_scoring_batch(
            self, items: list[dict[str, str]], chunk_size: int
        ) -> list[dict[str, str]]:
            encoded = self._tokenizer(["prompt"] * (len(items) * 3))
            self.model(**encoded)
            return [{"_system1_inference_origin": "real"} for _ in items]

    engine = Engine()
    try:
        reply = runner._run_timed_request(
            {
                "input": "batch_4",
                "boundary": "model_only_forward",
                "batch_size": 4,
                "runs": 2,
                "warmups": 5,
            },
            engine,
        )
    finally:
        from job_mcp.core.system1.factory import reset_engine

        reset_engine()
    assert engine.original_tokenizer.calls == 1
    assert engine.model.calls == 7
    assert len(reply["samples"]) == 2
    assert {sample["origin"] for sample in reply["samples"]} == {"real"}


def test_worker_cold_records_load_and_first_inference_without_warmup() -> None:
    runner = _load_runner()

    class Engine:
        def __init__(self) -> None:
            self.events: list[str] = []

        def load_model(self) -> Any:
            self.events.append("load")
            return SimpleNamespace(
                parameters=lambda: iter([SimpleNamespace(device="cpu")])
            )

        def predict_match_scoring_ensemble(
            self, *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            self.events.append("infer")
            return {"_system1_inference_origin": "real"}

    engine = Engine()
    reply = runner._run_timed_request(
        {
            "input": "cold_load",
            "boundary": "cold_load_first_inference",
            "runs": 1,
            "warmups": 0,
        },
        engine,
    )
    assert engine.events == ["load", "infer"]
    assert [sample["boundary"] for sample in reply["samples"]] == [
        "cold_load",
        "first_inference",
    ]
    assert reply["samples"][1]["origin"] == "real"


def test_worker_injects_retained_engine_and_explicit_profile_into_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    from job_mcp.core import api_client
    from job_mcp.core.system1 import factory

    seen: list[tuple[Any, Any]] = []
    engine = SimpleNamespace(
        load_model=lambda: SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device="cpu")])
        )
    )

    def fake_filter(
        jobs: list[Any], prefs: Any, profile: Any = None, **kwargs: Any
    ) -> list[Any]:
        seen.append((factory.get_system1_engine(), profile))
        for job in jobs:
            job._system1_inference_origin = "real"
        return jobs

    monkeypatch.setattr(api_client, "filter_jobs", fake_filter)
    try:
        runner._run_timed_request(
            {
                "input": "filter_jobs",
                "boundary": "filter_jobs",
                "runs": 1,
                "warmups": 5,
            },
            engine,
        )
    finally:
        factory.reset_engine()
    assert seen and seen[0][0] is engine
    assert seen[0][1] is not None
    assert seen[0][1].skills != ["Python"]


def test_quality_filters_mixed_447_record_holdout_before_inference_and_counting() -> None:
    runner = _load_runner()
    match_rows = [row for index in range(80) for row in _rows(f"state-{index}")]
    unrelated = [
        {"task": "role_classification", "state": f"other-{index}", "label": 0}
        for index in range(207)
    ]
    records = [*match_rows, *unrelated]

    class Engine:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def predict_match_scoring_batch(
            self, items: list[dict[str, str]]
        ) -> list[dict[str, Any]]:
            self.batch_sizes.append(len(items))
            return [
                {
                    "skill_match": 4,
                    "skill_confidence": 0.9,
                    "seniority_fit": 2,
                    "seniority_confidence": 0.8,
                    "recruiter_fit_probability": 0.7,
                    "_system1_inference_origin": "real",
                }
                for _ in items
            ]

    engine = Engine()
    metrics, origins = runner._quality_views(records, [], engine)
    assert origins == ["real"]
    assert engine.batch_sizes == [80, 80]
    assert metrics["source"] == {
        "total_record_count": 447,
        "match_scoring_record_count": 240,
    }
    assert metrics["views"]["primary"]["record_count"] == 240
    assert sum(
        item["count"]
        for item in metrics["views"]["primary"]["calibration"]["bins"]
    ) == 240


def test_quality_computes_complete_a_and_overlap_excluded_b_metrics() -> None:
    runner = _load_runner()
    records = _rows("overlap") + _rows("retained", (3, 0, 0))
    excluded = [hashlib.sha256(records[0]["state"].encode()).hexdigest()]

    class Engine:
        def predict_match_scoring_batch(
            self, items: list[dict[str, str]]
        ) -> list[dict[str, Any]]:
            return [
                {
                    "skill_match": 4,
                    "skill_confidence": 0.9,
                    "seniority_fit": 2,
                    "seniority_confidence": 0.8,
                    "recruiter_fit_probability": 0.7,
                    "_system1_inference_origin": "real",
                }
                for _ in items
            ]

    metrics, origins = runner._quality_views(records, excluded, Engine())
    assert origins == ["real"]
    assert metrics["views"]["primary"]["pair_count"] == 2
    assert metrics["views"]["sensitivity_b"]["pair_count"] == 1
    assert (
        metrics["views"]["primary"].keys() == metrics["views"]["sensitivity_b"].keys()
    )
    assert {"calibration", "seniority_mismatch"} <= metrics["views"][
        "sensitivity_b"
    ].keys()


def test_runtime_calibration_config_is_required_valid_and_matches_engine(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    engine = SimpleNamespace(model_name=str(tmp_path), temperature=0.75)

    with pytest.raises(RuntimeError, match="rl_agent_config.json"):
        runner._calibration_provenance(engine)

    config = tmp_path / "rl_agent_config.json"
    config.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="rl_agent_config.json"):
        runner._calibration_provenance(engine)

    config.write_text('{"calibrated_temperature": 0}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="calibrated_temperature"):
        runner._calibration_provenance(engine)

    config.write_text('{"calibrated_temperature": 0.8}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        runner._calibration_provenance(engine)

    config.write_text('{"calibrated_temperature": 0.75}', encoding="utf-8")
    assert runner._calibration_provenance(engine) == {
        "temperature": 0.75,
        "temperature_config_value": 0.75,
        "temperature_source": "rl_agent_config.json:calibrated_temperature",
        "temperature_config_sha256": canonical_file_checksum(config),
    }


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (lambda lanes: lanes.pop("model_forward"), "invalid benchmark lanes"),
        (
            lambda lanes: lanes.__setitem__("unexpected", dict(lanes["cold_load"])),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"].__setitem__("measured_count", 99),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["protocol"].__setitem__(
                "samples_per_run", 49
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"].__setitem__("run_ids", ["wrong"]),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"].pop("raw_samples"),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["raw_samples"][0].__setitem__(
                "warmup", True
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["raw_samples"][0].pop("input_id"),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["raw_samples"][0].__setitem__(
                "run_id", "wrong"
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["raw_samples"][0].__setitem__(
                "sample_index", False
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["raw_samples"][0].__setitem__(
                "process", True
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["raw_samples"][0].__setitem__(
                "process", 10**400
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["raw_samples"][0].__setitem__(
                "elapsed_ns", True
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["raw_samples"][0].__setitem__(
                "elapsed_ns", 10**400
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["raw_samples"][0].__setitem__(
                "elapsed_ns", 1.0
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["warm_ensemble"]["raw_samples"][0].__setitem__(
                "process", 10**400
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["protocol"].__setitem__(
                "run_count", True
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"].pop("percentile_resolution"),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["batch_4"]["metadata"].__setitem__("batch_size", 1),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["batch_4"]["metadata"].__setitem__("batch_size", True),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["batch_4"]["raw_samples"][0].__setitem__(
                "batch_size", True
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["batch_4"]["raw_samples"][0].__setitem__(
                "throughput_pairs_per_second", 1
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["batch_4"].__setitem__(
                "identity", "batch_1"
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["batch_4"].__setitem__(
                "boundary", "warm_ensemble"
            ),
            "invalid benchmark lanes",
        ),
        (
            lambda lanes: lanes["model_forward"]["summary"].__setitem__("mean", 0),
            "invalid benchmark lanes",
        ),
    ],
)
def test_publication_validation_rejects_malformed_lane_contract(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    expected_error: str,
) -> None:
    valid, artifacts = _valid_result(monkeypatch)
    mutate(valid["lanes"])
    assert expected_error in validate_publication(valid, artifacts)


def test_publication_validation_is_strict_for_nested_provenance_artifacts_model_and_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid, artifacts = _valid_result(monkeypatch)
    assert validate_publication(valid, artifacts, git_dirty=False) == []
    invalid_digest = {
        **valid,
        "artifacts": {**valid["artifacts"], "training": "not-a-digest"},
    }
    assert "invalid artifact checksum: training" in validate_publication(
        invalid_digest, artifacts
    )
    invalid_source = {**valid, "source": {"revision": "not-a-revision"}}
    assert "invalid source revision" in validate_publication(invalid_source, artifacts)
    invalid_model = {**valid, "model": {**valid["model"], "int8_linear_modules": 0}}
    assert "INT8 model inspection failed" in validate_publication(
        invalid_model, artifacts
    )
    missing_config = {
        **valid,
        "model": {
            key: value
            for key, value in valid["model"].items()
            if key != "temperature_config_sha256"
        },
    }
    assert "invalid model configuration schema" in validate_publication(
        missing_config, artifacts
    )
    mismatched_config = {
        **valid,
        "model": {**valid["model"], "temperature_config_value": 0.8},
    }
    assert "invalid model calibration configuration" in validate_publication(
        mismatched_config, artifacts
    )
    oversized_temperature = {
        **valid,
        "model": {**valid["model"], "temperature": 10**400},
    }
    assert "invalid model calibration configuration" in validate_publication(
        oversized_temperature, artifacts
    )
    for invalid_temperature in (
        True,
        float("nan"),
        float("inf"),
        float("-inf"),
        "0.75",
    ):
        # Each publication temperature field must route through the canonical
        # finite-number validator, never an unsafe float()/math.isfinite() cast.
        for temperature_key in ("temperature", "temperature_config_value"):
            unsafe_model = {**valid["model"], temperature_key: invalid_temperature}
            assert validate_publication(
                {**valid, "model": unsafe_model}, artifacts
            ), (temperature_key, invalid_temperature)
        paired = {
            **valid,
            "model": {
                **valid["model"],
                "temperature": invalid_temperature,
                "temperature_config_value": invalid_temperature,
            },
        }
        assert "invalid model calibration configuration" in validate_publication(
            paired, artifacts
        ), invalid_temperature
    assert "calibration config checksum mismatch" in validate_publication(
        valid, artifacts, expected_config_checksum="d" * 64
    )
    assert "quality source count mismatch" in validate_publication(
        valid,
        artifacts,
        expected_source_counts={
            "total_record_count": 447,
            "match_scoring_record_count": 240,
        },
    )
    invalid_threads = {
        **valid,
        "environment": {
            **valid["environment"],
            "torch_num_threads": {"requested": 4, "effective": 2},
        },
    }
    assert "invalid torch thread configuration" in validate_publication(
        invalid_threads, artifacts
    )
    disabled_quantization = {
        **valid,
        "model": {
            **valid["model"],
            "quantization": {"requested": False, "effective": "dynamic-int8"},
        },
    }
    assert "INT8 model inspection failed" in validate_publication(
        disabled_quantization, artifacts
    )
    invalid_quality = {
        **valid,
        "quality": {"inference_origins": ["real"], "metrics": {"views": {}}},
    }
    assert "invalid quality metrics schema" in validate_publication(
        invalid_quality, artifacts
    )


def test_quality_delta_is_derived_from_primary_and_sensitivity_views() -> None:
    from job_mcp.evaluation.benchmark import quality_delta_b_minus_a

    views = _quality_views()
    primary = deepcopy(views["primary"])
    sensitivity = deepcopy(views["sensitivity_b"])
    sensitivity["calibration"]["ece"] += 0.1
    sensitivity["seniority_mismatch"]["accuracy"] += 0.1
    sensitivity["seniority_mismatch"]["precision"] += 0.1
    sensitivity["seniority_mismatch"]["recall"] += 0.1
    sensitivity["seniority_mismatch"]["f1"] += 0.1
    sensitivity["seniority_mismatch"]["specificity"] += 0.1
    sensitivity["seniority_mismatch"]["balanced_accuracy"] += 0.1
    delta = quality_delta_b_minus_a({"primary": primary, "sensitivity_b": sensitivity})
    assert delta == {
        "ece": pytest.approx(0.1),
        "accuracy": pytest.approx(0.1),
        "precision": pytest.approx(0.1),
        "recall": pytest.approx(0.1),
        "f1": pytest.approx(0.1),
        "specificity": pytest.approx(0.1),
        "balanced_accuracy": pytest.approx(0.1),
    }


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda result: result["quality"]["metrics"].pop("delta_b_minus_a"),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["delta_b_minus_a"].__setitem__(
                "ece", 1.0
            ),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["delta_b_minus_a"].__setitem__(
                "ece", "0.0"
            ),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["delta_b_minus_a"].__setitem__(
                "ece", True
            ),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["delta_b_minus_a"].__setitem__(
                "ece", float("nan")
            ),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["delta_b_minus_a"].__setitem__(
                "ece", float("inf")
            ),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["delta_b_minus_a"].__setitem__(
                "ece", 10**400
            ),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["views"]["primary"].__setitem__(
                "pair_count", 10**400
            ),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["source"].__setitem__(
                "total_record_count", 10**400
            ),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["environment"]["torch_num_threads"].__setitem__(
                "requested", 10**400
            ),
            "invalid torch thread configuration",
        ),
        (
            lambda result: result["model"].__setitem__("int8_linear_modules", 10**400),
            "INT8 model inspection failed",
        ),
    ],
)
def test_publication_validation_rejects_invalid_quality_delta(
    monkeypatch: pytest.MonkeyPatch, mutate: Any, expected_error: str
) -> None:
    valid, artifacts = _valid_result(monkeypatch)
    mutate(valid)
    assert expected_error in validate_publication(valid, artifacts)


def test_publication_validation_rejects_invalid_audit_view_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = "invalid quality audit/view provenance"
    mismatch = "quality metrics/provenance view mismatch"
    mutations = [
        (
            lambda result: result["views"]["sensitivity_b"].__setitem__("pair_count", 2),
            provenance,
        ),
        (
            lambda result: result["views"]["sensitivity_b"].__setitem__("excluded_pairs", 1),
            provenance,
        ),
        (
            lambda result: result["views"]["sensitivity_b"].__setitem__("record_count", 6),
            provenance,
        ),
        (
            lambda result: result["quality"]["metrics"]["views"]["sensitivity_b"].__setitem__(
                "pair_count", 2
            ),
            mismatch,
        ),
        (lambda result: result["audit"].__setitem__("task_state_overlap_count", 4), provenance),
        (lambda result: result["audit"].__setitem__("exact_overlap_count", True), provenance),
        (
            lambda result: result["audit"].__setitem__(
                "overlap_state_sha256", ["f" * 64, "f" * 64]
            ),
            provenance,
        ),
        (
            lambda result: result["audit"].__setitem__(
                "duplicate_shared_state_clusters",
                [{"state_sha256": "f" * 64, "pair_count": 1}],
            ),
            provenance,
        ),
        (
            lambda result: result["audit"].__setitem__(
                "conflicting_labels",
                [
                    {
                        "task": "not-a-match-task",
                        "state_sha256": "f" * 64,
                        "labels": ["0", "1"],
                    }
                ],
            ),
            provenance,
        ),
        (lambda result: result["audit"].pop("overlap_state_sha256"), provenance),
        (lambda result: result["views"]["primary"].pop("remaining_checksum"), provenance),
    ]
    for index, (mutate, expected_error) in enumerate(mutations):
        valid, artifacts = _valid_result(monkeypatch)
        mutate(valid)
        errors = validate_publication(valid, artifacts)
        assert expected_error in errors, (index, errors)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda result: result["publication"].__setitem__("elapsed_seconds", True),
            "publication hard budget exceeded",
        ),
        (
            lambda result: result["publication"].__setitem__("elapsed_seconds", "1"),
            "publication hard budget exceeded",
        ),
        (
            lambda result: result["quality"]["metrics"]["views"]["primary"][
                "calibration"
            ].__setitem__("ece", float("nan")),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["views"]["primary"][
                "calibration"
            ]["bins"][0].__setitem__("accuracy", float("inf")),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["views"]["primary"][
                "seniority_mismatch"
            ].__setitem__("tp", True),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["views"]["primary"][
                "seniority_mismatch"
            ].__setitem__("fp", -1),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["views"]["primary"][
                "seniority_mismatch"
            ].__setitem__("fn", 1.5),
            "invalid quality metrics schema",
        ),
        (
            lambda result: result["quality"]["metrics"]["views"]["primary"][
                "seniority_mismatch"
            ].__setitem__("precision", float("-inf")),
            "invalid quality metrics schema",
        ),
    ],
)
def test_publication_validation_rejects_malformed_numeric_values(
    monkeypatch: pytest.MonkeyPatch, mutate: Any, expected_error: str
) -> None:
    valid, artifacts = _valid_result(monkeypatch)
    mutate(valid)
    assert expected_error in validate_publication(valid, artifacts)


def test_cli_clean_success_rechecks_dirty_state_and_artifacts_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _load_runner()
    holdout = tmp_path / "holdout.jsonl"
    training = tmp_path / "training.jsonl"
    holdout_rows = _rows("holdout") + [
        {"task": "role_classification", "state": "other", "label": 0}
    ]
    holdout.write_text(
        "\n".join(json.dumps(row) for row in holdout_rows), encoding="utf-8"
    )
    training.write_text(
        "\n".join(json.dumps(row) for row in _rows("training")), encoding="utf-8"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"model")
    config = model / "rl_agent_config.json"
    config.write_text('{"calibrated_temperature": 0.75}', encoding="utf-8")
    dirty_checks: list[bool] = []
    monkeypatch.setattr(
        runner, "_git_dirty", lambda: dirty_checks.append(False) or False
    )
    monkeypatch.setattr(runner, "_source_revision", lambda: "b" * 40)
    monkeypatch.setattr(
        runner,
        "dependency_versions",
        lambda: {"job-mcp": "0.1.0", "torch": "2.0.0", "transformers": "4.48.0"},
    )

    def fake_protocol(command: list[str], deadline_ns: int) -> dict[str, Any]:
        result, _ = _valid_result(monkeypatch)
        result["quality"]["metrics"]["source"] = {
            "total_record_count": 4,
            "match_scoring_record_count": 3,
        }
        result["model"]["temperature_config_sha256"] = canonical_file_checksum(
            config
        )
        return {
            key: result[key]
            for key in ("lanes", "quality", "environment")
        } | {"runtime_model": result["model"]}

    monkeypatch.setattr(runner, "execute_protocol", fake_protocol)
    assert (
        runner.main(
            [
                "--publish",
                "--holdout",
                str(holdout),
                "--training",
                str(training),
                "--model",
                str(model),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "success"
    assert len(dirty_checks) == 2
    assert output["artifacts"]["holdout"] == canonical_file_checksum(holdout)
    assert output["artifacts"]["training"] == canonical_file_checksum(training)


def test_cli_rejects_dirty_override_and_missing_artifacts_offline() -> None:
    script = Path(__file__).parents[2] / ".scripts" / "benchmark_milestone3.py"
    override = subprocess.run(
        [sys.executable, str(script), "--allow-dirty"],
        capture_output=True,
        text=True,
        check=False,
    )
    missing = subprocess.run(
        [sys.executable, str(script), "--publish", "--holdout", "missing"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert override.returncode == 2
    assert "unrecognized arguments: --allow-dirty" in override.stderr
    assert missing.returncode == 2
    assert "publication failed:" in missing.stderr


def test_protocol_constants_are_fixed() -> None:
    assert COLD_TIMEOUT_SECONDS == 120
    assert PUBLICATION_BUDGET_SECONDS == 75 * 60


def _metrics_view(name: str, hierarchy: str, pairs: int) -> dict[str, Any]:
    """Canonical multi-pair metrics view built only from the public helpers."""
    confidences: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    for _ in range(pairs):
        confidences.extend([0.9, 0.8, 0.7])
        predictions.extend([4, 2, 1])
        targets.extend([4, 2, 1])
    return {
        "name": name,
        "hierarchy": hierarchy,
        "pair_count": pairs,
        "record_count": pairs * 3,
        "calibration": calibration_summary(confidences, predictions, targets),
        "seniority_mismatch": binary_metrics([False] * pairs, [False] * pairs),
    }


def _consistent_result(
    monkeypatch: pytest.MonkeyPatch, latency_ns: int
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build a result whose every derived field comes from the given duration.

    The candidate latency is injected at the worker boundary, so
    ``execute_protocol`` recomputes each lane summary, percentile-resolution
    metadata, per-sample throughput, and throughput summary from it. A rejection
    therefore cannot be an artefact of stale metadata.
    """
    return _valid_result(monkeypatch, latency_ns=latency_ns)


def _hostile_latency_result(
    monkeypatch: pytest.MonkeyPatch, hostile: Any
) -> tuple[dict[str, Any], dict[str, str]]:
    """One fully consistent 3.5s result with a single hostile measured duration."""
    result, artifacts = _valid_result(monkeypatch, latency_ns=3_500_000_000)
    result["lanes"]["warm_ensemble"]["raw_samples"][0]["elapsed_ns"] = hostile
    return result, artifacts


def test_hostile_latency_is_rejected_before_any_float_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No OverflowError/TypeError may escape validation of a hostile duration."""
    for hostile in (10**400, -(10**400), None, "3500000000", 1.0, True, 0, -1):
        result, artifacts = _hostile_latency_result(monkeypatch, hostile)
        errors = validate_publication(result, artifacts)
        assert "invalid benchmark lanes" in errors, hostile


def test_publication_validation_accepts_realistic_multi_second_latencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durations must never inherit count bounds: real cold loads exceed 3s."""
    result, artifacts = _consistent_result(monkeypatch, 3_500_000_000)
    assert validate_publication(result, artifacts) == []
    cold = result["lanes"]["cold_load"]["raw_samples"]
    assert all(sample["elapsed_ns"] > benchmark.MAX_PUBLICATION_COUNT for sample in cold)
    batch_sample = result["lanes"]["batch_15"]["raw_samples"][0]
    assert batch_sample["elapsed_ns"] > benchmark.MAX_PUBLICATION_COUNT
    assert batch_sample["throughput_pairs_per_second"] == pytest.approx(
        15 * 1_000_000_000 / batch_sample["elapsed_ns"]
    )


def test_latency_ceiling_is_derived_from_the_publication_budget() -> None:
    assert benchmark.MAX_LATENCY_NS == PUBLICATION_BUDGET_SECONDS * 1_000_000_000
    assert benchmark.MAX_LATENCY_NS > COLD_TIMEOUT_SECONDS * 1_000_000_000
    assert benchmark.MAX_LATENCY_NS != benchmark.MAX_PUBLICATION_COUNT


@pytest.mark.parametrize("latency_ns", [3_500_000_000, benchmark.MAX_LATENCY_NS])
def test_latency_policy_accepts_every_consistent_duration_up_to_the_ceiling(
    monkeypatch: pytest.MonkeyPatch, latency_ns: int
) -> None:
    result, artifacts = _consistent_result(monkeypatch, latency_ns)
    assert validate_publication(result, artifacts) == []
    assert all(
        sample["elapsed_ns"] >= latency_ns
        for sample in result["lanes"]["model_forward"]["raw_samples"]
    )


def test_latency_ceiling_rejection_is_attributable_to_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the bound itself rejects, not incidental summary staleness."""
    over = benchmark.MAX_LATENCY_NS + 1
    result, artifacts = _consistent_result(monkeypatch, over)
    lane = result["lanes"]["model_forward"]
    assert lane["measured_count"] == len(lane["raw_samples"]) == 100
    assert lane["summary"] == benchmark.percentile_summary(
        [sample["elapsed_ns"] for sample in lane["raw_samples"]]
    )
    assert lane["percentile_resolution"]["sample_count"] == 100
    assert "invalid benchmark lanes" in validate_publication(result, artifacts)

    # A policy that accepts any positive int must accept the identical fixture,
    # so the rejection above can only come from the latency ceiling.
    monkeypatch.setattr(
        benchmark, "_latency_ns", lambda value: type(value) is int and value > 0
    )
    loosened, loosened_artifacts = _consistent_result(monkeypatch, over)
    assert validate_publication(loosened, loosened_artifacts) == []


@pytest.mark.parametrize(
    "lane_identity",
    ["cold_load", "model_forward", "warm_ensemble", "batch_4", "batch_15"],
)
@pytest.mark.parametrize(
    "invalid_process",
    [[], {}, "123", (1, 2), set(), True, 1.0, None, -1, 0, 10**400],
)
def test_process_field_fails_closed_before_hashing(
    monkeypatch: pytest.MonkeyPatch, lane_identity: str, invalid_process: Any
) -> None:
    """Unhashable process values must be rejected, never raise TypeError."""
    result, artifacts = _valid_result(monkeypatch)
    result["lanes"][lane_identity]["raw_samples"][0]["process"] = invalid_process
    errors = validate_publication(result, artifacts)
    assert "invalid benchmark lanes" in errors, (lane_identity, invalid_process)


def test_process_field_accepts_one_valid_pid_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One strictly positive integer PID per run must remain valid."""
    result, artifacts = _valid_result(monkeypatch)
    assert validate_publication(result, artifacts) == []
    processes = {
        sample["process"] for sample in result["lanes"]["batch_8"]["raw_samples"]
    }
    assert len(processes) == 2
    assert all(type(value) is int and value > 0 for value in processes)


@pytest.mark.parametrize(
    "invalid_throughput",
    [True, "1.0", float("nan"), float("inf"), float("-inf"), 10**400],
)
def test_batch_throughput_numeric_validation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, invalid_throughput: Any
) -> None:
    """A pathological throughput must produce an error, never raise OverflowError."""
    result, artifacts = _valid_result(monkeypatch)
    result["lanes"]["batch_8"]["raw_samples"][0][
        "throughput_pairs_per_second"
    ] = invalid_throughput
    errors = validate_publication(result, artifacts)
    assert errors is not None
    assert "invalid benchmark lanes" in errors, invalid_throughput


def test_batch_throughput_finite_mismatch_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, artifacts = _valid_result(monkeypatch)
    sample = result["lanes"]["batch_4"]["raw_samples"][0]
    assert sample["throughput_pairs_per_second"] != 0.5
    sample["throughput_pairs_per_second"] = 0.5
    assert "invalid benchmark lanes" in validate_publication(result, artifacts)


def test_batch_lane_thorough_summary_must_match_its_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, artifacts = _valid_result(monkeypatch)
    result["lanes"]["batch_1"]["throughput"]["mean"] += 1.0
    assert "invalid benchmark lanes" in validate_publication(result, artifacts)


@pytest.mark.parametrize(
    "invalid_source",
    [
        {},
        {"total_record_count": "x", "match_scoring_record_count": 3},
        {"total_record_count": None, "match_scoring_record_count": 3},
        {"total_record_count": True, "match_scoring_record_count": 3},
        {"total_record_count": -1, "match_scoring_record_count": 3},
        {"total_record_count": 10**400, "match_scoring_record_count": 3},
        {"total_record_count": 3, "match_scoring_record_count": 4},
        {"total_record_count": 3},
        None,
        "source",
        3,
    ],
)
def test_malformed_quality_source_is_rejected_without_raising(
    monkeypatch: pytest.MonkeyPatch, invalid_source: Any
) -> None:
    """P1-1: audit provenance must never raise KeyError/TypeError on bad JSON."""
    result, artifacts = _valid_result(monkeypatch)
    result["quality"]["metrics"]["source"] = invalid_source
    errors = validate_publication(result, artifacts)
    assert "invalid quality metrics schema" in errors
    assert "invalid quality audit/view provenance" in errors


def test_delta_mismatch_reports_its_own_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, artifacts = _valid_result(monkeypatch)
    result["quality"]["metrics"]["delta_b_minus_a"]["ece"] = 0.5
    assert "invalid quality metrics schema" in validate_publication(result, artifacts)


def test_audit_view_provenance_errors_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutations = [
        lambda result: result["views"]["sensitivity_b"].__setitem__("pair_count", 2),
        lambda result: result["views"]["sensitivity_b"].__setitem__("excluded_pairs", 1),
        lambda result: result["views"]["sensitivity_b"].__setitem__("record_count", 6),
        lambda result: result["audit"].__setitem__("task_state_overlap_count", 4),
        lambda result: result["audit"].__setitem__("exact_overlap_count", True),
        lambda result: result["audit"].__setitem__(
            "overlap_state_sha256", ["f" * 64, "f" * 64]
        ),
        lambda result: result["audit"].__setitem__("overlap_state_sha256", "f" * 64),
        lambda result: result["audit"].pop("overlap_state_sha256"),
        lambda result: result["views"]["primary"].pop("remaining_checksum"),
        lambda result: result["audit"].__setitem__(
            "conflicting_labels",
            [
                {"task": "match_scoring_skill", "state_sha256": "f" * 64, "labels": ["0", "1"]},
                {"task": "match_scoring_skill", "state_sha256": "f" * 64, "labels": ["0", "1", "2"]},
            ],
        ),
    ]
    for index, mutate in enumerate(mutations):
        result, artifacts = _valid_result(monkeypatch)
        mutate(result)
        assert "invalid quality audit/view provenance" in validate_publication(
            result, artifacts
        ), (index, validate_publication(result, artifacts))


def test_quality_metrics_provenance_mismatch_reports_its_own_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, artifacts = _valid_result(monkeypatch)
    result["views"]["primary"]["pair_count"] = 1
    result["quality"]["metrics"]["views"]["primary"]["pair_count"] = 2
    errors = validate_publication(result, artifacts)
    assert "quality metrics/provenance view mismatch" in errors


def test_producer_conflicts_round_trip_through_the_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2: real audit output must validate verbatim, without manual reshaping."""
    holdout = (
        _rows("state-a", labels=(4, 2, 1))
        + _rows("state-a", labels=(4, 5, 1))
        + _rows("state-b")
    )
    training = [
        # A third skill label plus a task outside the approved match-scoring set.
        {"task": "match_scoring_skill", "state": holdout[0]["state"], "label": 7},
        {"task": "other_task", "state": holdout[0]["state"], "label": 8},
    ]
    audit = audit_holdout_overlap(training, holdout)
    keys = [(entry["task"], entry["state_sha256"]) for entry in audit["conflicting_labels"]]
    assert len(keys) == len(set(keys))
    assert audit["conflicting_label_count"] == len(keys)
    assert all(task in REQUIRED_MATCH_SCORING_TASKS for task, _ in keys)

    views = primary_and_sensitivity_views(holdout, audit)
    match_records = [row for row in holdout if row["task"] in REQUIRED_MATCH_SCORING_TASKS]
    result, artifacts = _valid_result(monkeypatch)
    result["audit"] = audit
    result["views"] = views
    result["quality"]["metrics"] = {
        "source": {
            "total_record_count": len(holdout),
            "match_scoring_record_count": len(match_records),
        },
        "views": {
            "primary": _metrics_view(
                "primary_reproduction", "A-primary", views["primary"]["pair_count"]
            ),
            "sensitivity_b": _metrics_view(
                "overlap_excluded_sensitivity", "B-sensitivity", views["sensitivity_b"]["pair_count"]
            ),
        },
    }
    result["quality"]["metrics"]["delta_b_minus_a"] = quality_delta_b_minus_a(
        result["quality"]["metrics"]["views"]
    )
    assert validate_publication(result, artifacts) == []


def test_task_state_overlap_denominator_is_all_task_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlap counts every holdout row, not only match-scoring rows."""
    extra = [
        {"task": "other_task", "state": f"state-{index}", "label": index}
        for index in range(4)
    ]
    holdout = _rows("state-0") + extra
    training = [
        {"task": "other_task", "state": f"state-{index}", "label": index}
        for index in range(4)
    ]
    audit = audit_holdout_overlap(training, holdout)
    match_count = len([row for row in holdout if row["task"] in REQUIRED_MATCH_SCORING_TASKS])
    assert audit["task_state_overlap_count"] == 4
    assert audit["task_state_overlap_count"] > match_count

    views = primary_and_sensitivity_views(holdout, audit)
    result, artifacts = _valid_result(monkeypatch)
    result["audit"] = audit
    result["views"] = views
    result["quality"]["metrics"]["source"] = {
        "total_record_count": len(holdout),
        "match_scoring_record_count": match_count,
    }
    metrics = result["quality"]["metrics"]["views"]
    for key, hierarchy in (("primary", "A-primary"), ("sensitivity_b", "B-sensitivity")):
        metrics[key]["pair_count"] = views[key]["pair_count"]
        metrics[key]["record_count"] = views[key]["record_count"]
    assert validate_publication(result, artifacts) == []
