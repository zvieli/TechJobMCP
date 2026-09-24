#!/usr/bin/env python3
"""scripts/run_autonomous_job_hunt.py
-----------------------------------
End-to-end Autonomous Job Hunting Engine for TechJobMCP:
1. Scouts live Israeli tech jobs across public ATS boards (Comeet, Greenhouse, Lever, etc.).
2. Evaluates fit via System 1 Neural Matcher (Fine-Tuned LAYA / Hybrid matcher) with seniority capping.
3. Automatically triggers System 2 LLM Gateway for strong matches (score >= 80.0) to synthesize
   tailored resume highlights, customized cover letters, and technical interview questions.
4. Executes autonomous auto-apply via HybridApplicationDispatcher across all supported portals.
5. Captures and renders multi-layered submission receipts (DOM text, URL redirect, HTTP 200, screenshot proof).
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from job_mcp.core.api_client import calculate_match_score, extract_candidate_profile, filter_jobs
from job_mcp.core.application.dispatcher import HybridApplicationDispatcher
from job_mcp.core.application.ledger_service import ApplicationLedger
from job_mcp.core.application.tailoring import ApplicationPackage, generate_application_package
from job_mcp.core.system1.factory import get_system1_engine
from job_mcp.models.schemas import CandidateProfile, Job, JobPreferences
from job_mcp.sources.aggregator import JobAggregator
from job_mcp.sources.registry import create_default_registry

console = Console()
logging.basicConfig(level=logging.WARNING)


async def run_hunt(
    cv_path: str = "cv.pdf",
    min_score: float = 80.0,
    sources: Optional[List[str]] = None,
    dry_run: bool = False,
    max_applications: int = 3,
    output_receipts_file: str = "data/autonomous_run_receipts.json",
) -> List[Dict[str, Any]]:
    """Execute end-to-end autonomous job scout, tailoring, and application."""
    console.print(Panel(
        "[bold cyan]TechJobMCP Autonomous Job Hunt & Multi-Layer Auto-Apply Engine[/bold cyan]\n"
        f"[dim]System 1 Neural Triage + System 2 Application Tailoring + Multi-Layer Receipt Proof[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    # 1. Profile Extraction
    console.print("[cyan]Step 1: Extracting Candidate Profile from CV...[/cyan]")
    profile = extract_candidate_profile(cv_path)
    console.print(
        f"  Candidate: [bold]{profile.full_name or 'Lior Zvieli'}[/bold] | "
        f"Level: [bold yellow]{profile.seniority_level}[/bold yellow] | "
        f"Top Skills: [dim]{', '.join(profile.top_skills[:6])}[/dim]"
    )

    # 2. Live Job Scouting
    active_sources = sources or ["comeet", "greenhouse", "lever"]
    console.print(f"\n[cyan]Step 2: Scouting live positions across: [bold]{', '.join(active_sources)}[/bold]...[/cyan]")

    # Check if cached scout exists or fetch fresh
    cached_file = Path("data/scouted_jobs.json")
    scouted_jobs: List[Job] = []

    if cached_file.exists():
        try:
            with open(cached_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                scouted_jobs = [Job.model_validate(j) for j in data]
            console.print(f"  [green]Loaded {len(scouted_jobs)} cached scouted positions from {cached_file}.[/green]")
        except Exception as e:
            console.print(f"  [yellow]Notice reading cache ({e}); fetching live...[/yellow]")

    if not scouted_jobs:
        aggregator = JobAggregator(create_default_registry(), source_timeout=25.0)
        prefs = JobPreferences(
            cv_path=cv_path,
            location="Israel",
            tech_stack=profile.top_skills[:5],
            keywords=["Developer", "Engineer", "Python", "AI", "Full Stack"],
        )
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task("Fetching live listings...", total=None)
            scouted_jobs = await aggregator.fetch_all_jobs(
                sources=active_sources,
                preferences=prefs,
                profile=profile,
                limit_per_source=40,
                force_refresh=True,
            )
        console.print(f"  [green]Retrieved {len(scouted_jobs)} listings.[/green]")

    # 3. System 1 Scoring & Seniority Fit
    console.print(f"\n[cyan]Step 3: Neural Scoring with System 1 Engine (Target Score >= {min_score})...[/cyan]")
    prefs = JobPreferences(
        cv_path=cv_path,
        location="Israel",
        tech_stack=profile.top_skills[:5],
        keywords=["Developer", "Engineer", "Python", "React", "AI", "Full Stack"],
    )
    scored_jobs: List[Job] = []
    for job in scouted_jobs:
        score = calculate_match_score(job, prefs, profile=profile, enable_system1=True)
        job.match_score = score
        scored_jobs.append(job)

    scored_jobs.sort(key=lambda j: j.match_score or 0.0, reverse=True)
    strong_matches = [j for j in scored_jobs if (j.match_score or 0.0) >= min_score]

    console.print(f"  Total analyzed: [bold]{len(scored_jobs)}[/bold] | Strong matches (>= {min_score}): [bold green]{len(strong_matches)}[/bold green]")

    if not strong_matches:
        console.print(f"[yellow]No positions scored >= {min_score}. Highest position was '{scored_jobs[0].title}' at {scored_jobs[0].company} ({scored_jobs[0].match_score:.1f}).[/yellow]")
        return []

    # Display Top Matches
    match_table = Table(title="Candidate Top Matches (System 1 Scored)", box=box.SIMPLE_HEAVY)
    match_table.add_column("Score", style="bold green", width=8)
    match_table.add_column("Title", style="bold white", width=32)
    match_table.add_column("Company", style="cyan", width=18)
    match_table.add_column("Location", style="yellow", width=18)
    match_table.add_column("Source", style="magenta", width=12)

    for j in strong_matches[:5]:
        match_table.add_row(
            f"{j.match_score:.1f}",
            j.title[:30],
            j.company[:16],
            j.location[:16] or "Israel",
            j.source,
        )
    console.print(match_table)

    # 4. System 2 Tailoring & Auto-Apply Execution
    console.print(f"\n[bold cyan]Step 4: Autonomous Application & Verification Receipts (Max: {max_applications})[/bold cyan]")
    dispatcher = HybridApplicationDispatcher()
    receipts: List[Dict[str, Any]] = []

    target_jobs = strong_matches[:max_applications]

    for idx, target_job in enumerate(target_jobs, 1):
        console.print(f"\n[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]")
        console.print(f"[bold yellow][{idx}/{len(target_jobs)}] Processing Application: '{target_job.title}' at {target_job.company} (Score: {target_job.match_score:.1f})[/bold yellow]")

        # A. System 2 Tailoring
        console.print("  [cyan]Synthesizing tailored application package via System 2 LLM...[/cyan]")
        try:
            pkg = await generate_application_package(target_job, profile)
            console.print(f"  [green]✓ Custom Cover Letter Generated ({len(pkg.custom_cover_letter)} chars)[/green]")
            console.print(f"  [green]✓ Recruiter Pitch:[/green] [dim]\"{pkg.recruiter_pitch}\"[/dim]")
            console.print(f"  [green]✓ Generated {len(pkg.interview_prep_questions)} Interview Questions & Answering Strategies[/green]")
        except Exception as e:
            console.print(f"  [yellow]System 2 tailoring fallback used: {e}[/yellow]")
            pkg = ApplicationPackage(
                job_id=target_job.job_id,
                job_title=target_job.title,
                company=target_job.company,
                tailored_cv_highlights=[
                    f"Strong proficiency in {', '.join(profile.top_skills[:3])}",
                    f"Proven track record in autonomous software engineering",
                ],
                custom_cover_letter=f"Dear Hiring Team at {target_job.company},\n\nI am writing to express my strong enthusiasm for the {target_job.title} role. With my background in {', '.join(profile.top_skills[:4])}, I look forward to delivering immediate impact.\n\nBest regards,\n{profile.full_name or 'Lior Zvieli'}",
                recruiter_pitch=f"Full-stack and AI software engineer with core strengths in {', '.join(profile.top_skills[:3])}.",
                interview_prep_questions=[],
            )

        # B. Auto-Apply Execution
        app_profile = profile.model_copy()
        app_profile.cover_letter = pkg.custom_cover_letter

        if dry_run:
            console.print("  [bold yellow][DRY RUN] Previewing form mapping without submitting...[/bold yellow]")
            prev = await dispatcher.preview_application(target_job, app_profile, cv_path=cv_path)
            console.print(f"  Method: [bold]{prev.application_method}[/bold] | Fields mapped: {len(prev.fields_to_submit)}")
            receipt_entry = {
                "job_id": target_job.job_id,
                "title": target_job.title,
                "company": target_job.company,
                "source": target_job.source,
                "score": target_job.match_score,
                "status": "dry_run_preview",
                "preview_fields": list(prev.fields_to_submit.keys()),
                "cover_letter_snippet": pkg.custom_cover_letter[:150] + "...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            receipts.append(receipt_entry)
        else:
            console.print(f"  [cyan]Dispatching autonomous application to {target_job.source.upper()} portal...[/cyan]")
            apply_res = await dispatcher.execute_application(
                target_job,
                app_profile,
                cv_path=cv_path,
                force=True,  # Override auto apply toggle for interactive command
            )

            is_success = apply_res.get("success", False)
            status_text = apply_res.get("status", "unknown")
            receipt_meta = apply_res.get("receipt_details") or {}

            console.print(
                f"  Status: [{'green' if is_success else 'red'}][bold]{status_text.upper()}[/bold][/{'green' if is_success else 'red'}] | "
                f"Method: [bold]{apply_res.get('method', 'browser')}[/bold]"
            )

            if receipt_meta:
                console.print(Panel(
                    f"[bold white]Confirmation Type:[/bold white] [bold green]{receipt_meta.get('confirmation_type', 'CONFIRMED')}[/bold green]\n"
                    f"[bold white]Receipt Note:[/bold white] {receipt_meta.get('receipt_text', 'Application successfully received')}\n"
                    f"[bold white]Confirmation URL:[/bold white] {receipt_meta.get('confirmation_url') or target_job.apply_url or target_job.url}\n"
                    f"[bold white]Screenshot Proof:[/bold white] [dim]{receipt_meta.get('screenshot_path', 'None')}[/dim]\n"
                    f"[bold white]Submission Timestamp:[/bold white] {receipt_meta.get('timestamp')}",
                    title="Submission Receipt Proof",
                    border_style="green" if is_success else "yellow",
                    box=box.ROUNDED,
                ))

            receipt_entry = {
                "job_id": target_job.job_id,
                "title": target_job.title,
                "company": target_job.company,
                "source": target_job.source,
                "score": target_job.match_score,
                "success": is_success,
                "status": status_text,
                "receipt": receipt_meta or apply_res.get("receipt"),
                "tailored_package": pkg.model_dump(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            receipts.append(receipt_entry)

    # 5. Persist run receipts
    Path(output_receipts_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_receipts_file, "w", encoding="utf-8") as f:
        json.dump(receipts, f, indent=2, ensure_ascii=False)
    console.print(f"\n[bold green]✓ Hunt run complete. Stored {len(receipts)} receipt packages in {output_receipts_file}[/bold green]")

    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Job Hunt & Multi-Layer Auto-Apply Engine")
    parser.add_argument("--cv", default="cv.pdf", help="Path to CV document")
    parser.add_argument("--min-score", type=float, default=80.0, help="Minimum System 1 match score (default: 80.0)")
    parser.add_argument("--dry-run", action="store_true", help="Preview application packages without live submission")
    parser.add_argument("--limit", type=int, default=2, help="Max applications to submit (default: 2)")
    parser.add_argument("--sources", nargs="+", default=["comeet", "greenhouse", "lever"], help="Target ATS sources")
    parser.add_argument("--out", default="data/autonomous_run_receipts.json", help="Path to save audit receipts")

    args = parser.parse_args()
    asyncio.run(run_hunt(
        cv_path=args.cv,
        min_score=args.min_score,
        sources=args.sources,
        dry_run=args.dry_run,
        max_applications=args.limit,
        output_receipts_file=args.out,
    ))


if __name__ == "__main__":
    main()
