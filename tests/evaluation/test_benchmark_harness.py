"""Focused, artifact-free checks for the Milestone 3 benchmark harness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from job_mcp.evaluation import benchmark
from job_mcp.evaluation.benchmark import (
    COLD_TIMEOUT_SECONDS,
    PUBLICATION_BUDGET_SECONDS,
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
    protocol_observation_counts,
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
            },
            {
                "run": 0,
                "input": request["input"],
                "boundary": "first_inference",
                "elapsed_ns": 2,
                "origin": "real",
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
) -> tuple[dict[str, Any], dict[str, str]]:
    calls = 0

    def fake_call(
        command: list[str], request: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _fake_reply(request, calls)

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
    assert result["observations"] == protocol_observation_counts()
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
    assert any(sample["boundary"] == "cold_load" for sample in result["raw_samples"])
    assert any(
        sample["boundary"] == "first_inference" for sample in result["raw_samples"]
    )
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
            for key in ("observations", "raw_samples", "quality", "environment")
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
