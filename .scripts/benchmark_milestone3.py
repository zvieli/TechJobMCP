#!/usr/bin/env python3
"""Offline Milestone 3 benchmark runner with a fixed internal subprocess worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from job_mcp.evaluation.benchmark import (
    PUBLICATION_BUDGET_SECONDS,
    REQUIRED_MATCH_SCORING_TASKS,
    WARMUP_RUNS,
    audit_holdout_overlap,
    benchmark_input,
    binary_metrics,
    build_match_scoring_triples,
    calibration_summary,
    canonical_file_checksum,
    canonical_model_tree_checksum,
    dependency_versions,
    empty_manifest,
    environment_capture,
    execute_protocol,
    primary_and_sensitivity_views,
    source_revision,
    validate_publication,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Offline Milestone 3 benchmark protocol"
    )
    command.add_argument(
        "--publish",
        action="store_true",
        help="Explicitly run the bounded publication protocol",
    )
    command.add_argument(
        "--holdout", type=Path, default=Path("data/holdout/laya_val.jsonl")
    )
    command.add_argument(
        "--training", type=Path, default=Path("data/training/laya_train.jsonl")
    )
    command.add_argument("--model", type=Path, default=Path("data/models/laya-techjob"))
    command.add_argument(
        "--print-manifest",
        action="store_true",
        help="Print a path-free protocol manifest and exit",
    )
    command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return command


def _git_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return completed.returncode != 0 or bool(completed.stdout.strip())


def _source_revision() -> str:
    return source_revision()


def _engine_device(engine: Any, model: Any | None = None) -> str:
    model = model if model is not None else engine.load_model()
    if model is None:
        raise RuntimeError("model unavailable")
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _require_real(origins: list[Any], context: str) -> list[str]:
    if not origins or any(origin != "real" for origin in origins):
        raise RuntimeError(f"{context} returned non-real inference origin")
    return [str(origin) for origin in origins]


def _payload_digest(payload: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _profile_and_preferences() -> tuple[Any, Any]:
    from job_mcp.models.schemas import CandidateProfile, JobPreferences

    profile = CandidateProfile(
        skills=["Python", "FastAPI"],
        top_skills=["Python", "FastAPI"],
        primary_stack=["Python", "FastAPI"],
        target_roles=["Backend Engineer"],
        seniority_level="Mid",
        years_of_experience=3,
    )
    return profile, JobPreferences(tech_stack=["Python"])


def _prepare_operation(
    request: dict[str, Any], engine: Any
) -> Callable[[], tuple[list[str], int | None]]:
    """Prepare untimed inputs once and return the exact operation to measure."""
    from job_mcp.core.system1.factory import set_active_engine

    boundary = request["boundary"]
    payloads = request.get(
        "input_payloads", [request.get("input_payload", benchmark_input(0))]
    )
    payload_index = 0

    def next_payload() -> dict[str, str]:
        nonlocal payload_index
        payload = payloads[payload_index % len(payloads)]
        payload_index += 1
        return payload

    set_active_engine(engine)
    if boundary == "model_only_forward":
        model, tokenizer = engine.load_model(), engine._tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("model-only forward unavailable")
        size = int(request["batch_size"])
        encoded_by_digest = {}
        for payload in payloads:
            digest = _payload_digest(payload)
            if digest not in encoded_by_digest:
                encoded_by_digest[digest] = tokenizer(
                    [
                        f"{payload['job_title']}\n{payload['job_desc']}\n{payload['cv_text']}"
                    ]
                    * (size * 3),
                    padding=True,
                    return_tensors="pt",
                )
        active_payload_index = 0

        class Pretokenized:
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return encoded_by_digest[_payload_digest(payloads[active_payload_index])]

        engine._tokenizer = Pretokenized()
        elapsed_forward_ns: list[int] = []
        forward_started_ns = 0

        def before_forward(module: Any, inputs: Any) -> None:
            nonlocal forward_started_ns
            forward_started_ns = time.perf_counter_ns()

        def after_forward(module: Any, inputs: Any, output: Any) -> None:
            elapsed_forward_ns.append(time.perf_counter_ns() - forward_started_ns)

        model.register_forward_pre_hook(before_forward)
        model.register_forward_hook(after_forward)
        items_by_payload = [
            [dict(payload) for _ in range(size)] for payload in payloads
        ]

        def model_forward() -> tuple[list[str], int | None]:
            nonlocal active_payload_index
            elapsed_forward_ns.clear()
            active_payload_index = payload_index % len(items_by_payload)
            results = engine.predict_match_scoring_batch(
                items_by_payload[active_payload_index], chunk_size=size
            )
            next_payload()
            origins = _require_real(
                [result.get("_system1_inference_origin") for result in results],
                boundary,
            )
            if len(elapsed_forward_ns) != 1:
                raise RuntimeError(
                    "model-only boundary did not execute exactly one forward"
                )
            return origins, elapsed_forward_ns[0]

        return model_forward
    if boundary == "warm_ensemble":

        def warm_ensemble() -> tuple[list[str], int | None]:
            payload = next_payload()
            result = engine.predict_match_scoring_ensemble(
                payload["job_desc"], payload["cv_text"], payload["job_title"]
            )
            origins = _require_real([result.get("_system1_inference_origin")], boundary)
            return origins, None

        return warm_ensemble
    if boundary == "calculate_match_score":
        from job_mcp.core.api_client import calculate_match_score
        from job_mcp.models.schemas import Job

        profile, preferences = _profile_and_preferences()

        def match_score() -> tuple[list[str], int | None]:
            payload = next_payload()
            job = Job(
                job_id="benchmark",
                title=payload["job_title"],
                company="Fixture",
                location="Tel Aviv",
                description=payload["job_desc"],
                tech_stack=["Python", "FastAPI"],
            )
            calculate_match_score(
                job,
                preferences,
                profile=profile,
                enable_semantic=False,
                enable_system1=True,
            )
            return _require_real([job._system1_inference_origin], boundary), None

        return match_score
    if boundary == "filter_jobs":
        from job_mcp.core.api_client import filter_jobs
        from job_mcp.models.schemas import Job

        profile, preferences = _profile_and_preferences()

        def filtered_jobs() -> tuple[list[str], int | None]:
            payload = next_payload()
            jobs = [
                Job(
                    job_id=str(index),
                    title=payload["job_title"],
                    company="Fixture",
                    location="Tel Aviv",
                    description=payload["job_desc"],
                    tech_stack=["Python", "FastAPI"],
                )
                for index in range(3)
            ]
            scored = filter_jobs(
                jobs,
                preferences,
                profile=profile,
                enable_semantic=False,
                enable_system1=True,
            )
            origins = _require_real(
                [job._system1_inference_origin for job in scored], boundary
            )
            return origins, None

        return filtered_jobs
    raise RuntimeError("unsupported benchmark boundary")


def _run_timed_request(request: dict[str, Any], engine: Any) -> dict[str, Any]:
    """Run a request against one retained engine for this worker process."""
    boundary = request["boundary"]
    payloads = request.get(
        "input_payloads", [request.get("input_payload", benchmark_input(0))]
    )
    if boundary == "cold_load_first_inference":
        if request.get("warmups") != 0 or request.get("runs") != 1:
            raise RuntimeError("invalid cold protocol")
        load_started = time.perf_counter_ns()
        model = engine.load_model()
        load_elapsed = time.perf_counter_ns() - load_started
        if model is None:
            raise RuntimeError("model cold load failed")
        inference_started = time.perf_counter_ns()
        result = engine.predict_match_scoring_ensemble(
            payloads[0]["job_desc"],
            payloads[0]["cv_text"],
            payloads[0]["job_title"],
        )
        inference_elapsed = time.perf_counter_ns() - inference_started
        origins = _require_real(
            [result.get("_system1_inference_origin")], "cold first inference"
        )
        return {
            "status": "success",
            "device": _engine_device(engine, model),
            "pid": os.getpid(),
            "origins": origins,
            "samples": [
                {
                    "run": 0,
                    "input": request["input"],
                    "boundary": "cold_load",
                    "elapsed_ns": load_elapsed,
                    "input_payload_sha256": _payload_digest(payloads[0]),
                },
                {
                    "run": 0,
                    "input": request["input"],
                    "boundary": "first_inference",
                    "elapsed_ns": inference_elapsed,
                    "origin": origins[0],
                    "input_payload_sha256": _payload_digest(payloads[0]),
                },
            ],
        }
    if request.get("warmups") != WARMUP_RUNS:
        raise RuntimeError("warm workers require exactly five excluded warmups")
    operation_request = dict(request)
    operation_request["input_payloads"] = [payloads[0]] * WARMUP_RUNS + list(payloads)
    operation = _prepare_operation(operation_request, engine)
    for _ in range(WARMUP_RUNS):
        warmup_origins, _ = operation()
        _require_real(warmup_origins, "warmup")
    samples: list[dict[str, Any]] = []
    observed_origins: set[str] = set()
    for run in range(int(request["runs"])):
        started = time.perf_counter_ns()
        operation_origins, forward_elapsed = operation()
        origins = _require_real(operation_origins, "timed operation")
        elapsed = (
            forward_elapsed
            if forward_elapsed is not None
            else time.perf_counter_ns() - started
        )
        observed_origins.update(origins)
        samples.append(
            {
                "run": run,
                "input": request["input"],
                "boundary": boundary,
                "elapsed_ns": elapsed,
                "origin": origins[0],
                "input_payload_sha256": _payload_digest(payloads[run % len(payloads)]),
            }
        )
    return {
        "status": "success",
        "device": _engine_device(engine),
        "pid": os.getpid(),
        "origins": sorted(observed_origins),
        "samples": samples,
    }


def _state_digest(state: str) -> str:
    return hashlib.sha256(state.encode()).hexdigest()


def _quality_items(
    records: list[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, str]]]:
    triples = build_match_scoring_triples(records)
    items: list[dict[str, str]] = []
    for _, seniority, _ in triples:
        description, marker, cv = seniority["state"].partition("\nCandidate CV:\n")
        if not marker:
            raise RuntimeError("invalid holdout state")
        _, marker, description = description.partition("Job Description:\n")
        if not marker:
            raise RuntimeError("invalid holdout state")
        items.append(
            {
                "job_title": seniority.get("metadata", {}).get(
                    "job_title", "Unknown Role"
                ),
                "job_desc": description.strip(),
                "cv_text": cv.strip(),
            }
        )
    return triples, items


def _quality_metric_view(
    records: list[dict[str, Any]],
    engine: Any,
    *,
    name: str,
    hierarchy: str,
) -> tuple[dict[str, Any], list[str]]:
    triples, items = _quality_items(records)
    results = engine.predict_match_scoring_batch(items)
    if len(results) != len(triples):
        raise RuntimeError("quality inference count mismatch")
    origins = (
        _require_real(
            [result.get("_system1_inference_origin") for result in results], name
        )
        if results
        else []
    )
    confidences: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    for (skill, seniority, recruiter), result in zip(triples, results):
        confidences.extend(
            [
                result["skill_confidence"],
                result["seniority_confidence"],
                max(
                    result["recruiter_fit_probability"],
                    1 - result["recruiter_fit_probability"],
                ),
            ]
        )
        predictions.extend(
            [
                result["skill_match"],
                result["seniority_fit"],
                int(result["recruiter_fit_probability"] >= 0.5),
            ]
        )
        targets.extend(
            [int(skill["label"]), int(seniority["label"]), int(recruiter["label"])]
        )
    mismatch = binary_metrics(
        [result["seniority_fit"] == 0 for result in results],
        [int(seniority["label"]) == 0 for _, seniority, _ in triples],
    )
    return {
        "name": name,
        "hierarchy": hierarchy,
        "pair_count": len(triples),
        "record_count": len(records),
        "calibration": calibration_summary(confidences, predictions, targets),
        "seniority_mismatch": mismatch,
    }, origins


def _quality_views(
    records: list[dict[str, Any]], excluded_state_digests: list[str], engine: Any
) -> tuple[dict[str, Any], list[str]]:
    match_records = [
        row for row in records if row.get("task") in REQUIRED_MATCH_SCORING_TASKS
    ]
    excluded = set(excluded_state_digests)
    sensitivity = [
        row
        for row in match_records
        if _state_digest(str(row.get("state"))) not in excluded
    ]
    primary_metrics, primary_origins = _quality_metric_view(
        match_records,
        engine,
        name="primary_reproduction",
        hierarchy="A-primary",
    )
    sensitivity_metrics, sensitivity_origins = _quality_metric_view(
        sensitivity,
        engine,
        name="overlap_excluded_sensitivity",
        hierarchy="B-sensitivity",
    )
    return {
        "source": {
            "total_record_count": len(records),
            "match_scoring_record_count": len(match_records),
        },
        "views": {"primary": primary_metrics, "sensitivity_b": sensitivity_metrics},
    }, sorted(set(primary_origins + sensitivity_origins))


def _calibration_provenance(engine: Any) -> dict[str, Any]:
    config_path = Path(engine.model_name) / "rl_agent_config.json"
    if not config_path.is_file():
        raise RuntimeError("rl_agent_config.json is required")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("rl_agent_config.json is malformed") from error
    configured = config.get("calibrated_temperature") if isinstance(config, dict) else None
    if (
        isinstance(configured, bool)
        or not isinstance(configured, (int, float))
        or not math.isfinite(configured)
        or configured <= 0
    ):
        raise RuntimeError("calibrated_temperature must be a positive number")
    actual = engine.temperature
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(actual)
        or actual <= 0
        or not math.isclose(float(actual), float(configured), rel_tol=0, abs_tol=1e-12)
    ):
        raise RuntimeError("loaded temperature does not match calibrated_temperature")
    return {
        "temperature": float(actual),
        "temperature_config_value": float(configured),
        "temperature_source": "rl_agent_config.json:calibrated_temperature",
        "temperature_config_sha256": canonical_file_checksum(config_path),
    }


def _runtime_model_inspection(engine: Any) -> dict[str, Any]:
    import torch

    model = engine.load_model()
    if model is None:
        raise RuntimeError("model unavailable during runtime inspection")
    count = sum(
        isinstance(module, torch.nn.quantized.dynamic.Linear)
        for module in model.modules()
    )
    if count <= 0:
        raise RuntimeError("no dynamic INT8 Linear modules found")
    quantization_requested = os.getenv(
        "ENABLE_INT8_QUANTIZATION", "true"
    ).strip().lower() in {"true", "1", "yes"}
    return {
        "quantization": {
            "requested": quantization_requested,
            "effective": "dynamic-int8",
        },
        "int8_linear_modules": count,
        **_calibration_provenance(engine),
    }


def _runtime_environment(device: str) -> dict[str, Any]:
    import torch

    try:
        requested_threads = int(os.getenv("TORCH_NUM_THREADS", "4"))
    except ValueError as error:
        raise RuntimeError("TORCH_NUM_THREADS must be an integer") from error
    return environment_capture(
        device,
        torch_num_threads_requested=requested_threads,
        torch_num_threads_effective=torch.get_num_threads(),
    )


def _worker_quality(engine: Any) -> dict[str, Any]:
    records = _records(Path(os.environ["LAYA_HOLDOUT_PATH"]))
    training = _records(Path(os.environ["LAYA_TRAINING_PATH"]))
    audit = audit_holdout_overlap(training, records)
    metrics, origins = _quality_views(records, audit["overlap_state_sha256"], engine)
    device = _engine_device(engine)
    runtime_model = _runtime_model_inspection(engine)
    return {
        "metrics": metrics,
        "origins": origins,
        "device": device,
        "runtime_model": runtime_model,
        "runtime_environment": _runtime_environment(device),
    }


def worker_main() -> int:
    from job_mcp.core.system1.engine import LazyLayaEngine

    request = json.load(sys.stdin)
    engine = LazyLayaEngine(model_name=os.environ["LAYA_MODEL_PATH"])
    try:
        if request.get("mode") == "quality":
            quality = _worker_quality(engine)
            print(
                json.dumps(
                    {
                        "status": "success",
                        "device": quality["device"],
                        "pid": os.getpid(),
                        "samples": [],
                        "origins": quality["origins"],
                        "metrics": quality["metrics"],
                        "runtime_model": quality["runtime_model"],
                        "runtime_environment": quality["runtime_environment"],
                    }
                )
            )
            return 0
        print(json.dumps(_run_timed_request(request, engine)))
        return 0
    except Exception as error:  # noqa: BLE001 - worker must serialize all local failures
        print(json.dumps({"status": "error", "error": type(error).__name__}))
        return 2


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _artifact_checksums(holdout: Path, training: Path, model: Path) -> dict[str, str]:
    return {
        "holdout": canonical_file_checksum(holdout),
        "training": canonical_file_checksum(training),
        "model": canonical_model_tree_checksum(model)["sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.worker:
        return worker_main()
    if args.print_manifest:
        print(json.dumps(empty_manifest(), sort_keys=True))
        return 0
    if not args.publish:
        parser().error(
            "refusing benchmark execution without --publish; use --print-manifest for offline configuration"
        )
    if _git_dirty():
        print("publication failed: dirty git tree", file=sys.stderr)
        return 2
    if (
        not args.holdout.is_file()
        or not args.training.is_file()
        or not args.model.is_dir()
        or not (args.model / "rl_agent_config.json").is_file()
    ):
        print(
            "publication failed: required holdout, training, model, or calibration config artifacts are unavailable",
            file=sys.stderr,
        )
        return 2
    revision = _source_revision()
    artifacts = _artifact_checksums(args.holdout, args.training, args.model)
    started = time.monotonic_ns()
    environment_names = ("LAYA_MODEL_PATH", "LAYA_HOLDOUT_PATH", "LAYA_TRAINING_PATH")
    old_environment = {name: os.environ.get(name) for name in environment_names}
    os.environ["LAYA_MODEL_PATH"] = str(args.model)
    os.environ["LAYA_HOLDOUT_PATH"] = str(args.holdout)
    os.environ["LAYA_TRAINING_PATH"] = str(args.training)
    try:
        protocol = execute_protocol(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            started + PUBLICATION_BUDGET_SECONDS * 1_000_000_000,
        )
    except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        print(f"publication failed: {error}", file=sys.stderr)
        return 2
    finally:
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    holdout_records = _records(args.holdout)
    audit = audit_holdout_overlap(_records(args.training), holdout_records)
    runtime_model = protocol.pop("runtime_model")
    manifest = empty_manifest(artifacts)
    manifest["source"] = {"revision": revision}
    manifest["environment"]["dependencies"] = dependency_versions()
    manifest["model"] = runtime_model
    result = {
        **manifest,
        **protocol,
        "status": "success",
        "publication": {
            "elapsed_seconds": (time.monotonic_ns() - started) / 1_000_000_000
        },
        "audit": audit,
        "views": primary_and_sensitivity_views(holdout_records, audit),
    }

    # Publication acceptance is intentionally last: artifacts, revision, and
    # working-tree state are observed again after the potentially long run.
    final_artifacts = _artifact_checksums(args.holdout, args.training, args.model)
    final_revision = _source_revision()
    final_dirty = _git_dirty()
    result["publication"]["elapsed_seconds"] = (
        time.monotonic_ns() - started
    ) / 1_000_000_000
    final_config_path = args.model / "rl_agent_config.json"
    final_config_checksum = (
        canonical_file_checksum(final_config_path)
        if final_config_path.is_file()
        else None
    )
    expected_source_counts = {
        "total_record_count": len(holdout_records),
        "match_scoring_record_count": sum(
            row.get("task") in REQUIRED_MATCH_SCORING_TASKS for row in holdout_records
        ),
    }
    blockers = validate_publication(
        result,
        final_artifacts,
        git_dirty=final_dirty,
        expected_config_checksum=final_config_checksum,
        expected_source_counts=expected_source_counts,
    )
    if final_revision != revision:
        blockers.append("source revision changed during benchmark")
    if blockers:
        print(
            json.dumps(
                {"status": "rejected", "validation_blockers": blockers}, sort_keys=True
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
