#!/usr/bin/env python3
"""scripts/scout_real_jobs.py
--------------------------
Autonomous live job scouting tool for TechJobMCP.
Scouts live public job boards (Comeet, Greenhouse, Lever, etc.) using candidate's
actual CV (cv.pdf), scores all jobs with the active System 1 neural engine,
filters out seniority mismatches (e.g. senior/lead roles), and renders top opportunities.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from job_mcp.core.api_client import calculate_match_score, extract_candidate_profile, filter_jobs
from job_mcp.core.system1.factory import get_system1_engine
from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences
from job_mcp.sources.aggregator import JobAggregator
from job_mcp.sources.registry import create_default_registry

console = Console()


async def scout_live_jobs(
    cv_path: str = "cv.pdf",
    min_score: float = 65.0,
    limit: int = 15,
    location: Optional[str] = "Israel",
    sources: Optional[List[str]] = None,
    output_file: Optional[str] = "data/scouted_jobs.json",
) -> List[Job]:
    """Execute live job scout and return ranked matches."""
    console.print(Panel(
        "[bold cyan]TechJobMCP Live Job Scout & Neural Matcher[/bold cyan]\n"
        f"[dim]Analyzing real-world positions with System 1 Neural Filtering (Location: {location or 'Global'})[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    # 1. Candidate Profile Extraction
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("Extracting Candidate Profile from CV...", total=None)
        profile = extract_candidate_profile(cv_path)

    console.print(Panel(
        f"[bold white]Name:[/bold white] {profile.full_name or 'Candidate'}\n"
        f"[bold white]Seniority:[/bold white] [bold yellow]{profile.seniority_level}[/bold yellow]\n"
        f"[bold white]Top Skills:[/bold white] {', '.join(profile.top_skills[:10])}\n"
        f"[bold white]Target Roles:[/bold white] {', '.join(profile.target_roles[:6])}\n"
        f"[bold white]Location Target:[/bold white] {location or 'Any'}",
        title="Candidate Profile Context",
        border_style="blue",
        box=box.ROUNDED,
    ))

    # 2. Aggregating live jobs
    active_sources = sources or ["comeet", "greenhouse", "lever"]
    console.print(f"\n[cyan]Querying active live ATS sources:[/cyan] [bold]{', '.join(active_sources)}[/bold]...")

    aggregator = JobAggregator(create_default_registry(), source_timeout=25.0)

    prefs = JobPreferences(
        cv_path=cv_path,
        location=location,
        tech_stack=profile.top_skills[:5],
        keywords=["Developer", "Engineer", "Python", "React", "AI", "Full Stack"],
    )

    t0 = time.perf_counter()
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task(f"Fetching listings across {len(active_sources)} sources...", total=None)
        raw_jobs = await aggregator.fetch_all_jobs(
            sources=active_sources,
            preferences=prefs,
            profile=profile,
            limit_per_source=40,
            force_refresh=True,
        )
    fetch_duration = time.perf_counter() - t0
    console.print(f"[green]Retrieved {len(raw_jobs)} unique listings across sources in {fetch_duration:.1f}s.[/green]")

    if not raw_jobs:
        console.print("[yellow]No listings retrieved from specified sources. Check internet connection or source status.[/yellow]")
        return []

    # 3. Neural Scoring and Seniority Filtering
    console.print(f"[cyan]Scoring and calibrating {len(raw_jobs)} jobs with System 1 Neural Engine...[/cyan]")
    engine = get_system1_engine()
    engine_name = type(engine).__name__
    console.print(f"[dim]Active System 1 Engine: {engine_name}[/dim]")

    t1 = time.perf_counter()
    scored_jobs: List[Job] = []

    for job in raw_jobs:
        score = calculate_match_score(job, prefs, profile=profile, enable_system1=True)
        job.match_score = score
        scored_jobs.append(job)

    # Sort descending by match score
    scored_jobs.sort(key=lambda j: j.match_score or 0.0, reverse=True)
    scoring_duration = time.perf_counter() - t1
    console.print(f"[green]Scored {len(scored_jobs)} jobs in {scoring_duration:.2f}s ({scoring_duration / len(scored_jobs) * 1000:.1f}ms/job).[/green]\n")

    # Filter to qualified matches above min_score
    qualified_matches = [j for j in scored_jobs if (j.match_score or 0.0) >= min_score][:limit]

    # Render Results Table
    results_table = Table(
        title=f"Top Recommended Opportunities for {profile.full_name} (Min Score: {min_score})",
        box=box.ROUNDED,
        show_lines=True,
    )
    results_table.add_column("#", justify="center", style="bold")
    results_table.add_column("Score", justify="center", style="bold")
    results_table.add_column("Company & Role", style="bold white")
    results_table.add_column("Location & Mode", style="cyan")
    results_table.add_column("Matched Skills", style="green")
    results_table.add_column("Neural Fit & Reasons", style="dim")
    results_table.add_column("Apply Link", style="blue")

    for i, job in enumerate(qualified_matches, 1):
        score_val = job.match_score or 0.0
        score_badge = f"[bold green]{score_val:.1f}[/bold green]" if score_val >= 75.0 else f"[bold yellow]{score_val:.1f}[/bold yellow]"
        
        company_role = f"[bold]{job.title}[/bold]\n[dim]{job.company or 'Unknown Company'}[/dim]"
        loc_mode = f"{job.location or 'Israel'}\n[dim]{job.work_mode.value if hasattr(job.work_mode, 'value') else job.work_mode}[/dim]"
        matched = ", ".join(job.matched_skills[:5]) if job.matched_skills else "[dim]General match[/dim]"
        
        reasons = "\n".join(job.match_reasons[:2]) if job.match_reasons else ""
        apply_url = f"[link={job.url}]View / Apply[/link]" if job.url else "[dim]N/A[/dim]"

        results_table.add_row(
            str(i),
            score_badge,
            company_role,
            loc_mode,
            matched,
            reasons,
            apply_url,
        )

    console.print(results_table)

    # Save to disk for persistence
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_data = [
            {
                "id": j.job_id,
                "title": j.title,
                "company": j.company,
                "location": j.location,
                "work_mode": str(j.work_mode),
                "match_score": j.match_score,
                "matched_skills": j.matched_skills,
                "match_reasons": j.match_reasons,
                "url": j.url,
            }
            for j in qualified_matches
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
        console.print(f"[dim]Saved top {len(qualified_matches)} recommendations to {output_file}[/dim]")

    return qualified_matches


def main():
    parser = argparse.ArgumentParser(description="Scout live job opportunities with TechJobMCP")
    parser.add_argument("--cv", type=str, default="cv.pdf", help="Path to CV file")
    parser.add_argument("--location", type=str, default="Israel", help="Target location (e.g. Israel, Remote)")
    parser.add_argument("--min-score", type=float, default=60.0, help="Minimum match score threshold")
    parser.add_argument("--limit", type=int, default=15, help="Maximum number of matches to display")
    parser.add_argument("--sources", nargs="+", default=["comeet", "greenhouse", "lever"], help="Job sources to query")
    parser.add_argument("--out", type=str, default="data/scouted_jobs.json", help="Output JSON path")
    args = parser.parse_args()

    asyncio.run(scout_live_jobs(
        cv_path=args.cv,
        location=args.location,
        min_score=args.min_score,
        limit=args.limit,
        sources=args.sources,
        output_file=args.out,
    ))


if __name__ == "__main__":
    main()
