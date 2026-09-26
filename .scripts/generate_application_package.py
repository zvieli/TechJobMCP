#!/usr/bin/env python3
"""scripts/generate_application_package.py
-----------------------------------------
Generates tailored application assets (CV highlights, cover letter, recruiter pitch,
and interview questions) for a target scouted job using System 2 deep reasoning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from job_mcp.core.api_client import extract_candidate_profile
from job_mcp.core.application.tailoring import generate_application_package
from job_mcp.models.schemas import Job, WorkMode

console = Console()


async def run_tailoring(
    cv_path: str = "cv.pdf",
    job_id: Optional[str] = None,
    scouted_jobs_file: str = "data/scouted_jobs.json",
) -> None:
    profile = extract_candidate_profile(cv_path)

    # Load scouted jobs
    jobs_path = Path(scouted_jobs_file)
    if not jobs_path.exists():
        console.print(f"[red]Scouted jobs file '{scouted_jobs_file}' not found. Run scout_real_jobs.py first![/red]")
        return

    with open(jobs_path, "r", encoding="utf-8") as f:
        scouted = json.load(f)

    if not scouted:
        console.print("[yellow]No scouted jobs found in file.[/yellow]")
        return

    target_data = None
    if job_id:
        for j in scouted:
            if j.get("id") == job_id:
                target_data = j
                break
        if not target_data:
            console.print(f"[red]Job ID '{job_id}' not found in {scouted_jobs_file}. Using #1 top job.[/red]")
            target_data = scouted[0]
    else:
        target_data = scouted[0]

    job = Job(
        job_id=target_data.get("id", "job_1"),
        title=target_data.get("title", "Software Engineer"),
        company=target_data.get("company", "Tech Company"),
        location=target_data.get("location", "Israel"),
        work_mode=WorkMode.HYBRID,
        tech_stack=target_data.get("matched_skills", ["Python", "React", "AI"]),
        description=f"Position at {target_data.get('company')} for {target_data.get('title')}. Matched skills: {', '.join(target_data.get('matched_skills', []))}.",
        url=target_data.get("url"),
    )

    console.print(Panel(
        f"[bold white]Target Role:[/bold white] {job.title} @ [bold cyan]{job.company}[/bold cyan]\n"
        f"[bold white]Location:[/bold white] {job.location} | [bold white]Score:[/bold white] [bold green]{target_data.get('match_score', 'N/A')}[/bold green]\n"
        f"[bold white]Candidate:[/bold white] {profile.full_name} ({profile.seniority_level})",
        title="[bold yellow]System 2 Application Tailoring & Interview Prep[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    ))

    with console.status("[bold green]Generating structured application package with System 2...[/bold green]"):
        pkg = await generate_application_package(job, profile)

    # 1. Highlights
    hl_table = Table(title="Tailored CV Highlights for Resume / Application Form", box=box.ROUNDED)
    hl_table.add_column("#", justify="center", style="bold")
    hl_table.add_column("Bullet Point", style="white")
    for i, hl in enumerate(pkg.tailored_cv_highlights, 1):
        hl_table.add_row(str(i), hl)
    console.print(hl_table)

    # 2. Recruiter Pitch
    console.print(Panel(
        pkg.recruiter_pitch,
        title="[bold cyan]Recruiter Cold Outreach / LinkedIn Elevator Pitch[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    # 3. Custom Cover Letter
    console.print(Panel(
        Markdown(pkg.custom_cover_letter),
        title="[bold magenta]Tailored Motivation & Cover Letter[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    ))

    # 4. Interview Prep Questions
    prep_table = Table(title="Anticipated Interview Questions & Response Strategy", box=box.ROUNDED, show_lines=True)
    prep_table.add_column("Topic", style="cyan", width=20)
    prep_table.add_column("Anticipated Question", style="bold white", width=40)
    prep_table.add_column("Recommended Strategy & Points to Highlight", style="green", width=50)

    for q in pkg.interview_prep_questions:
        prep_table.add_row(q.topic, q.question, q.recommended_strategy)

    console.print(prep_table)

    # Save to file
    out_file = Path("data/application_package.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(pkg.model_dump(), f, indent=2, ensure_ascii=False)
    console.print(f"[dim]Saved application package to {out_file}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Generate System 2 tailored application package")
    parser.add_argument("--cv", type=str, default="cv.pdf", help="Path to CV file")
    parser.add_argument("--job-id", type=str, default=None, help="Target job ID from scouted jobs")
    parser.add_argument("--file", type=str, default="data/scouted_jobs.json", help="Path to scouted jobs JSON")
    args = parser.parse_args()

    asyncio.run(run_tailoring(cv_path=args.cv, job_id=args.job_id, scouted_jobs_file=args.file))


if __name__ == "__main__":
    main()
