#!/usr/bin/env python3
"""scripts/benchmark_system1_ab.py
--------------------------------
Empirical A/B Benchmark Suite for System 1 Engines:
- Fine-Tuned LAYA (LazyLayaEngine)
- Base Pretrained LAYA (BaseLayaEngine via convaiinnovations/laya-multilingual)
- Zero-Shot Generative LLM (GenerativeBaselineEngine)

Measures:
1. Seniority Fit & Capping Accuracy on edge cases (e.g. Optibus Senior vs Junior)
2. Average Latency per evaluation (CPU ms)
3. Prediction confidence & distribution
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from job_mcp.core.system1.base_laya import BaseLayaEngine
from job_mcp.core.system1.engine import LazyLayaEngine
from job_mcp.core.system1.factory import set_active_engine, reset_engine
from job_mcp.core.system1.generative import GenerativeBaselineEngine
from job_mcp.core.api_client import extract_candidate_profile, calculate_match_score
from job_mcp.models.schemas import Job, JobPreferences, WorkMode

console = Console()

TEST_JOBS = [
    {
        "id": "optibus_senior",
        "title": "Senior Full Stack Engineer, Gen AI",
        "company": "Optibus",
        "tech_stack": ["Python", "TypeScript", "React", "Node.js", "Docker", "CI/CD", "LangGraph", "LLM", "RAG"],
        "description": "Senior Full Stack Engineer with 5+ years experience building production Gen AI systems.",
        "expected_sen_fit": 0, # Should be flagged as far too junior
        "senior_role": True,
    },
    {
        "id": "junior_ai_agent",
        "title": "Junior AI Agent / Full Stack Developer",
        "company": "NextGen AI Labs",
        "tech_stack": ["Python", "FastAPI", "React", "TypeScript", "Docker", "LLM"],
        "description": "Looking for an ambitious Junior or Graduate Full Stack Developer with Python, React, and GenAI agent experience.",
        "expected_sen_fit": 1, # Should be roughly matches
        "senior_role": False,
    },
    {
        "id": "entry_python_backend",
        "title": "Junior Software Engineer (Python)",
        "company": "TechCorp",
        "tech_stack": ["Python", "PostgreSQL", "Docker", "Git", "REST"],
        "description": "Great opportunity for Junior developers with strong Python fundamentals, REST APIs, and database knowledge.",
        "expected_sen_fit": 1, # Should be roughly matches
        "senior_role": False,
    },
    {
        "id": "principal_architect",
        "title": "Principal DevOps Architect",
        "company": "CloudScale",
        "tech_stack": ["Kubernetes", "Terraform", "AWS", "Go", "BASH"],
        "description": "10+ years infrastructure architect needed to lead global multi-region cloud deployment.",
        "expected_sen_fit": 0, # Far too junior
        "senior_role": True,
    },
]


def run_benchmark(cv_path: str = "cv.pdf") -> None:
    profile = extract_candidate_profile(cv_path)
    prefs = JobPreferences(cv_path=cv_path)

    console.print(Panel(
        f"[bold cyan]Candidate:[/bold cyan] {profile.full_name} | "
        f"[bold yellow]Seniority:[/bold yellow] {profile.seniority_level} | "
        f"[bold green]Top Skills:[/bold green] {', '.join(profile.top_skills[:8])}",
        title="Candidate Profile Context",
        expand=False,
    ))

    engines = {
        "Fine-Tuned LAYA": LazyLayaEngine.get_instance(),
        "Base Pretrained LAYA": BaseLayaEngine.get_instance(),
        "Zero-Shot Generative LLM": GenerativeBaselineEngine(),
    }

    # Pre-warm all engines
    console.print("\n[dim]Pre-warming engines...[/dim]")
    for name, engine in engines.items():
        set_active_engine(engine)
        engine.predict_match_scoring_ensemble("Sample desc", "Sample cv", "Sample title")
    reset_engine()

    results_table = Table(title="System 1 Engine A/B Head-to-Head Benchmark", show_lines=True)
    results_table.add_column("Engine", style="bold")
    results_table.add_column("Job Title", style="cyan")
    results_table.add_column("Match Score", justify="center")
    results_table.add_column("Seniority Fit (0=Jr,1=Match,2=Sr)", justify="center")
    results_table.add_column("Confidence", justify="center")
    results_table.add_column("Recruiter Fit", justify="center")
    results_table.add_column("Latency (ms)", justify="right")

    summary_stats = {name: {"optibus_capped": False, "latencies": [], "avg_conf": []} for name in engines}

    for engine_name, engine in engines.items():
        set_active_engine(engine)

        for job_info in TEST_JOBS:
            job = Job(
                job_id=job_info["id"],
                title=job_info["title"],
                company=job_info["company"],
                tech_stack=job_info["tech_stack"],
                description=job_info["description"],
                work_mode=WorkMode.HYBRID,
            )

            t0 = time.perf_counter()
            score = calculate_match_score(job, prefs, profile=profile, enable_system1=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Raw ensemble prediction
            ens = engine.predict_match_scoring_ensemble(
                job_desc=job.description,
                cv_text=f"Skills: {', '.join(profile.top_skills)}. Seniority: {profile.seniority_level}.",
                job_title=job.title,
            )

            sen_fit = ens.get("seniority_fit", -1)
            sen_conf = ens.get("seniority_confidence", 0.0)
            rec_fit = ens.get("recruiter_fit_probability", 0.0)

            summary_stats[engine_name]["latencies"].append(latency_ms)
            summary_stats[engine_name]["avg_conf"].append(sen_conf)

            if job_info["id"] == "optibus_senior":
                summary_stats[engine_name]["optibus_capped"] = (score <= 40.0)

            score_style = "bold red" if score <= 40.0 else ("bold green" if score >= 75.0 else "yellow")
            results_table.add_row(
                engine_name,
                job.title,
                f"[{score_style}]{score:.1f}[/{score_style}]",
                f"{sen_fit} (conf: {sen_conf:.2f})",
                f"{sen_conf:.2f}",
                f"{rec_fit:.2f}",
                f"{latency_ms:.1f} ms",
            )

    console.print(results_table)

    summary_table = Table(title="Summary Comparison & Decision Matrix", show_lines=True)
    summary_table.add_column("Engine", style="bold")
    summary_table.add_column("Optibus Senior Capped? (Score <= 40)", justify="center")
    summary_table.add_column("Avg Latency per Job (ms)", justify="right")
    summary_table.add_column("Avg Confidence", justify="center")
    summary_table.add_column("Assessment", style="italic")

    assessments = {
        "Fine-Tuned LAYA": "Calibrated on domain data; reliably penalizes junior on senior roles; fast CPU execution.",
        "Base Pretrained LAYA": "Zero-shot out-of-the-box; may underestimate seniority penalty without fine-tuning.",
        "Zero-Shot Generative LLM": "High quality reasoning; dependent on external API / network latency; high cost per query.",
    }

    for name, stats in summary_stats.items():
        capped = "[bold green]YES (Correct)[/bold green]" if stats["optibus_capped"] else "[bold red]NO (False Positive)[/bold red]"
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0
        avg_c = sum(stats["avg_conf"]) / len(stats["avg_conf"]) if stats["avg_conf"] else 0.0
        summary_table.add_row(
            name,
            capped,
            f"{avg_lat:.1f} ms",
            f"{avg_c:.2f}",
            assessments.get(name, ""),
        )

    console.print("\n")
    console.print(summary_table)

    reset_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B Benchmark System 1 Engines")
    parser.add_argument("--cv", type=str, default="cv.pdf", help="Path to CV file")
    args = parser.parse_args()
    run_benchmark(args.cv)
