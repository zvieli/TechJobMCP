"""Small, dependency-free helpers for reproducible model evaluation."""

from .benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    binary_metrics,
    build_match_scoring_triples,
    calibration_summary,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "binary_metrics",
    "build_match_scoring_triples",
    "calibration_summary",
]
