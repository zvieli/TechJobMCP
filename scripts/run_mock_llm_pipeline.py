#!/usr/bin/env python3
"""HireMeTech Visual CLI Runner Script.

Autonomous job hunting LLM agent pipeline simulator with Rich terminal output,
step-by-step reasoning traces, and interactive summary dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from job_mcp.testing.mock_llm import MockLLMAgent, PipelineResult, StepTrace


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the CLI argument parser."""
    default_cv = "lior_zvieli_cv.pdf" if os.path.exists("lior_zvieli_cv.pdf") else None

    parser = argparse.ArgumentParser(
        prog="run_mock_llm_pipeline",
        description="HireMeTech Autonomous Job Search LLM Agent Pipeline Visual Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--cv",
        type=str,
        default=default_cv,
        help="Path to candidate CV file (PDF, DOCX, TXT, TEX)",
    )
    parser.add_argument(
        "--stack",
        type=str,
        default="Python,FastAPI,AI,Agentic AI,LangGraph,React,TypeScript,Docker",
        help="Comma-separated target technology stack keywords",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="Senior,Lead,5+ years,10+ years",
        help="Comma-separated exclusion keywords to filter out",
    )
    parser.add_argument(
        "--top-tier",
        type=int,
        default=85,
        dest="top_tier",
        help="Minimum match score threshold for Top-Tier jobs (score >= top-tier)",
    )
    parser.add_argument(
        "--strong-match",
        type=int,
        default=70,
        dest="strong_match",
        help="Minimum match score threshold for Strong Match jobs (score >= strong-match)",
    )
    parser.add_argument(
        "--disqualify-threshold",
        type=int,
        default=50,
        dest="disqualify_threshold",
        help="Score threshold below which jobs are dismissed/deleted",
    )
    parser.add_argument(
        "--auto-apply",
        dest="auto_apply",
        action="store_true",
        default=True,
        help="Enable 2-step auto-apply (stage preview + confirm) for top-tier jobs (default: True)",
    )
    parser.add_argument(
        "--no-auto-apply",
        dest="auto_apply",
        action="store_false",
        help="Disable auto-apply for top-tier jobs",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["autonomous", "supervised", "mock"],
        default="autonomous",
        help="Server operation mode (default: autonomous)",
    )
    parser.add_argument(
        "--work-mode",
        type=str,
        choices=["remote", "hybrid", "onsite", "any"],
        default=None,
        help="Work mode preference filter",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Geographic location filter (e.g. 'Tel Aviv', 'Israel', 'Remote')",
    )
    parser.add_argument(
        "--min-salary",
        type=int,
        default=None,
        help="Minimum desired annual salary threshold",
    )
    parser.add_argument(
        "--target-roles",
        type=str,
        default=None,
        help="Comma-separated target role titles (e.g. 'AI Engineer,Full Stack Developer')",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help="Additional comma-separated keywords",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Force fresh scraping from live job sources bypassing local cache",
    )
    parser.add_argument(
        "--remote-url",
        type=str,
        default=None,
        help="Optional live MCP server URL (e.g. 'http://localhost:8000/mcp') to connect to over HTTP/SSE",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON serialization of PipelineResult at the end",
    )
    parser.add_argument(
        "--inspect",
        "--show-jobs",
        dest="inspect",
        action="store_true",
        default=True,
        help="Render detailed Job Insight Cards for Top-Tier and Strong Match jobs (default: True)",
    )
    parser.add_argument(
        "--no-inspect",
        dest="inspect",
        action="store_false",
        help="Disable rendering detailed Job Insight Cards",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Display detailed payload data and raw response dictionaries for each step",
    )

    return parser


def parse_comma_separated(val: Optional[str]) -> list[str]:
    """Parse comma-separated string into cleaned list of non-empty items."""
    if not val:
        return []
    return [item.strip() for item in val.split(",") if item.strip()]


def summarize_observation(tool_name: str, args: dict[str, Any], response: dict[str, Any]) -> str:
    """Create a concise, human-readable summary of a tool execution observation."""
    if not response.get("success", False):
        err_msg = response.get("message") or response.get("error_code") or "Tool execution error"
        return f"[bold red]FAILED:[/bold red] {err_msg}"

    data = response.get("data")

    if tool_name == "set_operation_mode":
        mode = data.get("mode") if isinstance(data, dict) else args.get("mode", "unknown")
        return f"Operation mode successfully configured to [bold green]{mode}[/bold green]."

    if tool_name == "list_job_sources":
        if isinstance(data, dict):
            sources = data.get("sources", [])
            names = [s.get("name") or s.get("source_id", "unknown") if isinstance(s, dict) else str(s) for s in sources]
            return f"Discovered [bold cyan]{len(names)}[/bold cyan] platforms: {', '.join(names)}"
        return f"Platforms discovered: {data}"

    if tool_name == "get_job_matches":
        if isinstance(data, list):
            sources_count: dict[str, int] = {}
            for j in data:
                if isinstance(j, dict):
                    src = j.get("source") or "unknown"
                    sources_count[src] = sources_count.get(src, 0) + 1
            src_summary = ", ".join(f"{k}: {v}" for k, v in sources_count.items()) or "cached/live"
            return f"Aggregated [bold cyan]{len(data)}[/bold cyan] job listings ({src_summary})"
        return f"Job listings retrieved: {data}"

    if tool_name == "filter_jobs_by_preferences":
        if isinstance(data, list):
            top_cnt = sum(1 for j in data if isinstance(j, dict) and float(j.get("match_score") or 0) >= 85)
            strong_cnt = sum(
                1 for j in data if isinstance(j, dict) and 70 <= float(j.get("match_score") or 0) < 85
            )
            return (
                f"Scored [bold cyan]{len(data)}[/bold cyan] matching jobs "
                f"([green]🏆 Top-Tier (85+): {top_cnt}[/green], [yellow]⭐ Strong (70-84): {strong_cnt}[/yellow])"
            )
        return f"Scored jobs: {data}"

    if tool_name == "bookmark_job":
        jid = args.get("job_id", "unknown")
        return f"Job [bold yellow]{jid}[/bold yellow] saved to bookmarks [green]✓[/green]"

    if tool_name == "auto_apply_job":
        jid = args.get("job_id", "unknown")
        company = ""
        if isinstance(data, dict):
            company = data.get("company", "")
        company_str = f" at {company}" if company else ""
        return f"Generated application preview for [bold yellow]{jid}[/bold yellow]{company_str} (Staged for confirmation [yellow]⏳[/yellow])"

    if tool_name == "confirm_auto_apply":
        jid = args.get("job_id", "unknown")
        conf_id = ""
        if isinstance(data, dict):
            conf_id = data.get("confirmation_id", "")
        conf_str = f" [dim](ID: {conf_id})[/dim]" if conf_id else ""
        return f"Application submitted successfully for [bold yellow]{jid}[/bold yellow]{conf_str} [bold green]✓[/bold green]"

    if tool_name == "delete_job":
        jid = args.get("job_id", "unknown")
        return f"Dismissed/deleted disqualified job [bold yellow]{jid}[/bold yellow] [dim]✓[/dim]"

    msg = response.get("message")
    if msg:
        return msg
    return str(data) if data is not None else "Operation completed successfully."


def render_header(
    console: Console,
    args: argparse.Namespace,
    tech_stack: list[str],
    exclude_keywords: list[str],
) -> None:
    """Render a header banner and configuration summary."""
    title_text = Text("🚀 HireMeTech - Autonomous Job Search Agent Pipeline", style="bold cyan")

    config_table = Table(box=box.SIMPLE_HEAD, show_header=False, pad_edge=False)
    config_table.add_column("Key", style="bold white", width=22)
    config_table.add_column("Value", style="cyan")

    config_table.add_row(
        "Candidate CV",
        f"[green]{args.cv}[/green]" if args.cv and os.path.exists(args.cv) else (f"[yellow]{args.cv} (not found)[/yellow]" if args.cv else "[dim]None (tech stack matching)[/dim]"),
    )
    config_table.add_row("Tech Stack Target", ", ".join(tech_stack) if tech_stack else "[dim]None[/dim]")
    config_table.add_row("Seniority Exclusions", ", ".join(exclude_keywords) if exclude_keywords else "[dim]None[/dim]")
    config_table.add_row(
        "Score Thresholds",
        f"[green]Top-Tier: ≥{args.top_tier}[/green]  |  [yellow]Strong Match: ≥{args.strong_match}[/yellow]  |  [dim]Disqualify: <{args.disqualify_threshold}[/dim]",
    )
    config_table.add_row(
        "Safety Auto-Apply",
        "[bold green]Enabled (2-Step Staged Preview & Confirm)[/bold green]" if args.auto_apply else "[bold yellow]Disabled (Bookmarks Only)[/bold yellow]",
    )
    config_table.add_row(
        "Job Insight Cards",
        "[bold green]Enabled (Detailed Match & Skill Cards)[/bold green]" if getattr(args, "inspect", True) else "[dim]Disabled (--no-inspect)[/dim]",
    )
    config_table.add_row(
        "Server Connection",
        f"[bold magenta]Remote MCP ({args.remote_url})[/bold magenta]" if args.remote_url else f"[bold blue]In-Memory Execution (Mode: {args.mode})[/bold blue]",
    )

    if args.work_mode or args.location:
        filters = []
        if args.work_mode:
            filters.append(f"Work Mode: {args.work_mode}")
        if args.location:
            filters.append(f"Location: {args.location}")
        config_table.add_row("Additional Filters", "  |  ".join(filters))

    header_panel = Panel(
        config_table,
        title=title_text,
        subtitle="[dim]Powered by FastMCP, Multi-Source Aggregator & MockLLMAgent[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print()
    console.print(header_panel)
    console.print()


def render_step_trace(console: Console, step: StepTrace, verbose: bool = False) -> None:
    """Render a single pipeline execution step trace."""
    # Tool call argument string
    arg_items = [f"{k}={v!r}" for k, v in step.arguments.items() if k != "ctx"]
    args_str = ", ".join(arg_items)
    if len(args_str) > 80:
        args_str = args_str[:77] + "..."

    obs_summary = summarize_observation(step.tool_name, step.arguments, step.response)

    # Step panel content
    step_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
    step_table.add_column("Icon", width=4)
    step_table.add_column("Label", style="bold", width=14)
    step_table.add_column("Content")

    if step.thought:
        step_table.add_row("🤖", "[cyan]Thought[/cyan]", f"[italic]{step.thought}[/italic]")

    step_table.add_row(
        "🛠️",
        "[green]Tool Call[/green]",
        f"[bold yellow]{step.tool_name}[/bold yellow]({args_str})",
    )
    step_table.add_row(
        "⏱️",
        "[dim]Duration[/dim]",
        f"[bold cyan]{step.duration_ms:.1f} ms[/bold cyan]",
    )
    step_table.add_row("📥", "[blue]Observation[/blue]", obs_summary)

    step_panel = Panel(
        step_table,
        title=f"[bold white]Step {step.step_number}[/bold white] — [yellow]{step.tool_name}[/yellow]",
        title_align="left",
        border_style="dim blue",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(step_panel)

    if verbose:
        detail_json = json.dumps(
            {"arguments": step.arguments, "response": step.response},
            indent=2,
            default=str,
        )
        console.print(
            Panel(
                Syntax(detail_json, "json", theme="monokai", line_numbers=False),
                title=f"[dim]Step {step.step_number} Payload Details[/dim]",
                border_style="dim",
                padding=(0, 1),
            )
        )
    console.print()


def _format_work_mode(mode_val: Any) -> str:
    """Format work mode cleanly for display."""
    if not mode_val:
        return "N/A"
    if hasattr(mode_val, "value"):
        mode_val = mode_val.value
    s = str(mode_val).replace("WorkMode.", "").strip().capitalize()
    return s


def render_job_insight_card(
    console: Console,
    job: dict[str, Any],
    result: PipelineResult,
    tier_label: str = "Top-Tier Match",
    border_style: str = "green",
) -> None:
    """Render a detailed Job Insight Card as a styled Rich Panel."""
    jid = job.get("job_id", "")
    title = job.get("title", "Untitled Role")
    company = job.get("company", "Unknown Company")
    score = float(job.get("match_score") or 0.0)
    seniority = job.get("seniority_level") or "Junior"
    work_mode_str = _format_work_mode(job.get("work_mode"))
    location = job.get("location") or "N/A"
    salary = job.get("salary_range")

    # 1. Source(s) Badges
    raw_sources = job.get("sources") or ([job.get("source")] if job.get("source") else [])
    source_badges: list[str] = []
    for s in raw_sources:
        s_lower = str(s).lower()
        if "comeet" in s_lower:
            source_badges.append("[bold magenta][Comeet (Direct ATS)][/bold magenta]")
        elif "hireme" in s_lower:
            source_badges.append("[bold blue][HireMeTech][/bold blue]")
        elif "alljobs" in s_lower:
            source_badges.append("[bold yellow][AllJobs][/bold yellow]")
        else:
            source_badges.append(f"[bold cyan][{s}][/bold cyan]")
    sources_str = " ".join(source_badges) if source_badges else "[dim]Unknown[/dim]"

    # 2. Match Score & Seniority Details Line
    score_style = "bold green" if score >= 85 else "bold yellow"
    details_parts = [
        f"Score: [{score_style}]{score:.0f}/100[/{score_style}]",
        f"Level: [cyan]{seniority}[/cyan]",
        f"Work Mode: [cyan]{work_mode_str}[/cyan]",
        f"Location: [cyan]{location}[/cyan]",
    ]
    if salary:
        details_parts.append(f"Salary: [green]{salary}[/green]")
    fit_line = "  |  ".join(details_parts)

    # 3. Match Reasons
    reasons = job.get("match_reasons") or []
    if reasons:
        reasons_str = "\n".join(f"• {r}" for r in reasons)
    else:
        tech_list = job.get("tech_stack") or []
        tech_str = ", ".join(tech_list[:5]) if tech_list else "General engineering match"
        reasons_str = f"• High alignment with target technology stack ({tech_str})"

    # 4. Matched and Missing Skills
    matched_skills = job.get("matched_skills") or job.get("tech_stack") or []
    missing_skills = job.get("missing_skills") or []

    matched_badges = (
        " ".join(f"[bold green][{s}][/bold green]" for s in matched_skills)
        if matched_skills
        else "[dim]None listed[/dim]"
    )
    missing_badges = (
        " ".join(f"[dim][{s}][/dim]" for s in missing_skills)
        if missing_skills
        else ""
    )

    # 5. Role Summary
    summary = job.get("description_summary") or job.get("description") or "No description available."
    if len(summary) > 280 and not job.get("description_summary"):
        from job_mcp.core.api_client import generate_description_summary
        summary = generate_description_summary(summary) or (summary[:277] + "...")

    # 6. Application Link
    link = job.get("apply_url") or job.get("url") or "N/A"
    if link != "N/A":
        link_str = f"[underline cyan]{link}[/underline cyan]"
    else:
        link_str = "[dim]N/A[/dim]"

    # 7. Auto-Apply Status
    if jid in result.confirmed_apply_ids:
        apply_status = "[bold green]✅ Applied & Confirmed[/bold green]"
    elif jid in result.staged_apply_ids:
        apply_status = "[bold yellow]⏳ Staged Preview (Pending Confirmation)[/bold yellow]"
    elif jid in result.bookmarked_job_ids:
        apply_status = "[bold cyan]🔖 Bookmarked for Review[/bold cyan]"
    else:
        apply_status = "[dim]Unapplied[/dim]"

    # Card Table
    card_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1), expand=True)
    card_table.add_column("Icon & Field", style="bold white", width=22)
    card_table.add_column("Content")

    card_table.add_row("🏷️  Source(s)", sources_str)
    card_table.add_row("🎯  Match Details", fit_line)
    card_table.add_row("💡  Match Reasons", reasons_str)
    card_table.add_row("🛠️  Matched Skills", matched_badges)
    if missing_badges:
        card_table.add_row("   Missing Skills", missing_badges)
    card_table.add_row("📝  Role Summary", summary)
    card_table.add_row("🔗  Application Link", link_str)
    card_table.add_row("⚡  Auto-Apply Status", apply_status)

    panel_title = f"📋 [bold white]{title}[/bold white] [dim]@[/dim] [bold cyan]{company}[/bold cyan] [dim]({jid})[/dim]"
    card_panel = Panel(
        card_table,
        title=panel_title,
        title_align="left",
        subtitle=f"[{border_style}]{tier_label} (Score: {score:.0f}/100)[/{border_style}]",
        subtitle_align="right",
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(card_panel)
    console.print()


def render_summary_dashboard(
    console: Console | PipelineResult,
    result: Optional[PipelineResult | Console] = None,
    top_tier_threshold: int = 85,
    strong_match_threshold: int = 70,
    inspect_jobs: bool = True,
) -> None:
    """Render the comprehensive final summary dashboard and optional Job Insight Cards."""
    if isinstance(console, PipelineResult):
        if isinstance(result, Console):
            console, result = result, console
        else:
            actual_result = console
            console = Console()
            result = actual_result
    elif result is None:
        raise ValueError("PipelineResult must be provided to render_summary_dashboard")

    console.print(
        Panel(
            Text("📊 Pipeline Execution Summary Dashboard", style="bold white on blue", justify="center"),
            box=box.DOUBLE,
            style="blue",
        )
    )
    console.print()

    # 1. Top-Tier Jobs Table
    console.print(
        f"[bold green]🏆 Top-Tier Matches (Score ≥ {top_tier_threshold})[/bold green] "
        f"[dim]({len(result.top_tier_jobs)} jobs found)[/dim]"
    )

    if result.top_tier_jobs:
        top_table = Table(box=box.ROUNDED, header_style="bold green", show_lines=True, expand=True)
        top_table.add_column("Score", justify="center", style="bold", ratio=1)
        top_table.add_column("Job Title", style="bold white", ratio=4)
        top_table.add_column("Company", style="cyan", ratio=3)
        top_table.add_column("Location / Mode", ratio=3)
        top_table.add_column("Source(s)", ratio=2)
        top_table.add_column("Auto-Apply Status", justify="center", ratio=3)

        for job in result.top_tier_jobs:
            jid = job.get("job_id", "")
            score = float(job.get("match_score") or 0.0)
            title = job.get("title", "Unknown")
            company = job.get("company", "Unknown")
            location = job.get("location") or "N/A"
            work_mode_str = _format_work_mode(job.get("work_mode"))
            loc_mode = f"{location}\n[dim]({work_mode_str})[/dim]"

            sources = job.get("sources") or ([job.get("source")] if job.get("source") else [])
            sources_str = ", ".join(sources) if sources else "Unknown"

            # Determine application status
            if jid in result.confirmed_apply_ids:
                apply_status = "[bold green]✅ Applied & Confirmed[/bold green]"
            elif jid in result.staged_apply_ids:
                apply_status = "[bold yellow]⏳ Staged Preview[/bold yellow]"
            elif jid in result.bookmarked_job_ids:
                apply_status = "[cyan]⭐ Bookmarked[/cyan]"
            else:
                apply_status = "[dim]Unapplied[/dim]"

            score_style = "bold green" if score >= 90 else "green"
            top_table.add_row(
                f"[{score_style}]{score:.0f}[/{score_style}]",
                title,
                company,
                loc_mode,
                sources_str,
                apply_status,
            )
        console.print(top_table)
    else:
        console.print("[italic dim]No jobs reached the Top-Tier threshold score in this run.[/italic dim]")

    console.print()

    # 2. Strong Match Jobs Table
    console.print(
        f"[bold yellow]⭐ Strong Matches (Score {strong_match_threshold}–{top_tier_threshold - 1})[/bold yellow] "
        f"[dim]({len(result.strong_match_jobs)} jobs found)[/dim]"
    )

    if result.strong_match_jobs:
        strong_table = Table(box=box.ROUNDED, header_style="bold yellow", show_lines=True, expand=True)
        strong_table.add_column("Score", justify="center", style="bold", ratio=1)
        strong_table.add_column("Job Title", style="bold white", ratio=4)
        strong_table.add_column("Company", style="cyan", ratio=3)
        strong_table.add_column("Location / Mode", ratio=3)
        strong_table.add_column("Source(s)", ratio=2)
        strong_table.add_column("Bookmark Status", justify="center", ratio=3)

        for job in result.strong_match_jobs:
            jid = job.get("job_id", "")
            score = float(job.get("match_score") or 0.0)
            title = job.get("title", "Unknown")
            company = job.get("company", "Unknown")
            location = job.get("location") or "N/A"
            work_mode_str = _format_work_mode(job.get("work_mode"))
            loc_mode = f"{location}\n[dim]({work_mode_str})[/dim]"

            sources = job.get("sources") or ([job.get("source")] if job.get("source") else [])
            sources_str = ", ".join(sources) if sources else "Unknown"

            b_status = "[bold cyan]🔖 Bookmarked[/bold cyan]" if jid in result.bookmarked_job_ids else "[dim]—[/dim]"

            strong_table.add_row(
                f"[yellow]{score:.0f}[/yellow]",
                title,
                company,
                loc_mode,
                sources_str,
                b_status,
            )
        console.print(strong_table)
    else:
        console.print("[italic dim]No jobs in the Strong Match score range in this run.[/italic dim]")

    console.print()

    # 3. Detailed Job Insight Cards (Rendered by default for Top-Tier and Strong Matches)
    if inspect_jobs:
        inspectable_jobs_count = len(result.top_tier_jobs) + len(result.strong_match_jobs)
        if inspectable_jobs_count > 0:
            console.print(
                f"[bold cyan]🔍 Detailed Job Insight Cards ({inspectable_jobs_count} priority jobs)[/bold cyan]"
            )
            console.print()
            for job in result.top_tier_jobs:
                render_job_insight_card(
                    console,
                    job,
                    result,
                    tier_label="🏆 Top-Tier Match",
                    border_style="green",
                )
            for job in result.strong_match_jobs:
                render_job_insight_card(
                    console,
                    job,
                    result,
                    tier_label="⭐ Strong Match",
                    border_style="yellow",
                )

    # 4. Overall KPI Metrics Table
    kpi_table = Table(box=box.ROUNDED, header_style="bold white on dark_blue", show_lines=True, expand=True)
    kpi_table.add_column("Metric", style="bold", ratio=4)
    kpi_table.add_column("Value", justify="right", style="bold cyan", ratio=3)
    kpi_table.add_column("Description", style="dim", ratio=5)

    kpi_table.add_row(
        "Total Pipeline Steps",
        str(len(result.steps)),
        "Agent reasoning thoughts and tool invocations",
    )
    kpi_table.add_row(
        "Total Jobs Fetched",
        str(result.total_jobs_fetched),
        f"Aggregated across {len(result.sources_found)} platform(s)",
    )
    kpi_table.add_row(
        "Top-Tier Matches (85+)",
        f"[green]{len(result.top_tier_jobs)}[/green]",
        "Target matches prioritized for auto-application",
    )
    kpi_table.add_row(
        "Strong Matches (70-84)",
        f"[yellow]{len(result.strong_match_jobs)}[/yellow]",
        "Secondary strong matches prioritized for bookmarking",
    )
    kpi_table.add_row(
        "Applications Submitted",
        f"[bold green]{len(result.confirmed_apply_ids)}[/bold green]",
        "Safety 2-step previewed and confirmed applications",
    )
    kpi_table.add_row(
        "Total Bookmarked",
        f"[cyan]{len(result.bookmarked_job_ids)}[/cyan]",
        "Listings saved for user review in portal",
    )
    kpi_table.add_row(
        "Disqualified Removed",
        f"[red]{len(result.deleted_job_ids)}[/red]",
        "Seniority exclusions or score below threshold",
    )
    kpi_table.add_row(
        "Total Execution Time",
        f"{result.execution_time_ms / 1000.0:.2f} s ({result.execution_time_ms:.1f} ms)",
        "End-to-end multi-source pipeline latency",
    )

    status_badge = "[bold green]PIPELINE SUCCESS ✓[/bold green]" if result.success else "[bold red]PIPELINE FAILED ✗[/bold red]"
    kpi_table.add_row("Overall Status", status_badge, "Final execution health indicator")

    console.print(
        Panel(
            kpi_table,
            title="[bold cyan]📈 Execution Performance & Pipeline KPIs[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    console.print()


def seed_mock_jobs_in_cache() -> None:
    """Pre-seed realistic multi-source jobs into global JobCache for deterministic mock mode."""
    from job_mcp.main import _get_cache
    from job_mcp.models.schemas import Job, WorkMode

    cache = _get_cache()
    mock_jobs = [
        Job(
            job_id="hmt-top-1",
            title="AI Application Engineer (FastAPI & LangGraph)",
            company="TechNova Labs",
            source="hiremetech",
            sources=["hiremetech", "comeet"],
            work_mode=WorkMode.REMOTE,
            tech_stack=[
                "Python",
                "FastAPI",
                "AI",
                "Agentic AI",
                "LangGraph",
                "React",
                "TypeScript",
                "Docker",
                "Git",
                "Linux",
                "LLM",
                "NLP",
                "LangChain",
                "Azure",
                "Pandas",
                "Scikit-Learn",
                "Node.js",
                "TailwindCSS",
            ],
            description="Building agentic workflows and full-stack MCP applications with Python, FastAPI, LangGraph, React, TypeScript, and Docker.",
            salary_range="$140,000 - $165,000",
            location="Tel Aviv / Remote",
            url="https://hireme.tech/jobs/hmt-top-1",
        ),
        Job(
            job_id="cmt-top-2",
            title="Autonomous Agents Software Developer (Python & AI)",
            company="Cognitive Dynamics",
            source="comeet",
            sources=["comeet"],
            work_mode=WorkMode.HYBRID,
            tech_stack=[
                "Python",
                "FastAPI",
                "AI",
                "Agentic AI",
                "LangGraph",
                "Docker",
                "TypeScript",
                "React",
                "Git",
                "Linux",
                "LLM",
                "NLP",
                "LangChain",
                "Azure",
            ],
            description="Develop autonomous agents and FastAPI backend microservices for enterprise automation using Python, React, and TypeScript.",
            salary_range="$135,000 - $150,000",
            location="Tel Aviv",
            apply_url="https://app.comeet.com/jobs/cognitivedynamics/cmt-top-2",
        ),
        Job(
            job_id="cmt-strong-1",
            title="Python & React Developer (AI Platforms)",
            company="CloudWorks Software",
            source="comeet",
            sources=["comeet"],
            work_mode=WorkMode.REMOTE,
            tech_stack=[
                "Python",
                "FastAPI",
                "AI",
                "React",
                "TypeScript",
                "Docker",
                "Git",
                "Linux",
                "LLM",
                "LangChain",
            ],
            description="Developing scalable AI web applications with Python, FastAPI backend, and React/TypeScript frontend.",
            salary_range="$125,000 - $138,000",
            location="Remote",
            apply_url="https://app.comeet.com/jobs/cloudworks/cmt-strong-1",
        ),
        Job(
            job_id="alljobs-strong-2",
            title="Python Software Developer (FastAPI & Docker)",
            company="InnoTech Solutions",
            source="alljobs",
            sources=["alljobs"],
            work_mode=WorkMode.HYBRID,
            tech_stack=[
                "Python",
                "AI",
                "FastAPI",
                "Docker",
                "Git",
                "Linux",
                "Pandas",
                "TypeScript",
                "React",
                "LLM",
            ],
            description="Join our team building data pipelines, FastAPI REST services, Docker containers, and AI application tools.",
            salary_range="₪32,000 - ₪38,000 / mo",
            location="Herzliya",
            url="https://www.alljobs.co.il/job/alljobs-strong-2",
        ),
        Job(
            job_id="hmt-disq-1",
            title="Senior Solutions Architect (10+ years)",
            company="Legacy Systems",
            source="hiremetech",
            sources=["hiremetech"],
            work_mode=WorkMode.ONSITE,
            tech_stack=["Java", "Spring", "Oracle"],
            description="Senior architect with 10+ years experience in legacy enterprise architectures.",
            salary_range="$180,000",
            location="Haifa",
            url="https://hireme.tech/jobs/hmt-disq-1",
        ),
        Job(
            job_id="cmt-disq-2",
            title="WordPress Theme Designer",
            company="MediaPress",
            source="comeet",
            sources=["comeet"],
            work_mode=WorkMode.ONSITE,
            tech_stack=["PHP", "WordPress"],
            description="WordPress developer designing themes and layouts.",
            salary_range="₪14,000 / mo",
            location="Ramat Gan",
            apply_url="https://app.comeet.com/jobs/mediapress/cmt-disq-2",
        ),
    ]
    cache.update(mock_jobs)


async def execute_cli_pipeline(
    args: argparse.Namespace,
    console: Optional[Console] = None,
) -> PipelineResult:
    """Execute the MockLLMAgent pipeline according to parsed CLI arguments."""
    if console is None:
        console = Console()

    tech_stack = parse_comma_separated(args.stack)
    exclude_keywords = parse_comma_separated(args.exclude)
    target_roles = parse_comma_separated(args.target_roles)
    keywords = parse_comma_separated(args.keywords)

    if not args.json:
        render_header(console, args, tech_stack, exclude_keywords)

    def on_step(trace: StepTrace) -> None:
        if not args.json:
            render_step_trace(console, trace, verbose=args.verbose)

    mode_to_set = "autonomous" if args.mode == "mock" else args.mode

    # Remote MCP Client or In-Memory Execution
    if args.remote_url:
        from fastmcp.client import Client

        async with Client(args.remote_url) as client:
            agent = MockLLMAgent(mcp_server=client, cv_path=args.cv)
            result = await agent.run_pipeline(
                tech_stack=tech_stack,
                exclude_keywords=exclude_keywords,
                cv_path=args.cv,
                top_tier_threshold=args.top_tier,
                strong_match_threshold=args.strong_match,
                auto_apply=args.auto_apply,
                target_roles=target_roles if target_roles else None,
                work_mode=args.work_mode if args.work_mode != "any" else None,
                location=args.location,
                min_salary=args.min_salary,
                keywords=keywords if keywords else None,
                force_refresh=args.force_refresh,
                disqualify_threshold=args.disqualify_threshold,
                mode=mode_to_set,
                step_callback=on_step,
            )
    else:
        from job_mcp.main import _get_cache
        if args.mode == "mock" or (not _get_cache().get_all() and not args.force_refresh):
            seed_mock_jobs_in_cache()

        agent = MockLLMAgent(cv_path=args.cv)
        result = await agent.run_pipeline(
            tech_stack=tech_stack,
            exclude_keywords=exclude_keywords,
            cv_path=args.cv,
            top_tier_threshold=args.top_tier,
            strong_match_threshold=args.strong_match,
            auto_apply=args.auto_apply,
            target_roles=target_roles if target_roles else None,
            work_mode=args.work_mode if args.work_mode != "any" else None,
            location=args.location,
            min_salary=args.min_salary,
            keywords=keywords if keywords else None,
            force_refresh=args.force_refresh,
            disqualify_threshold=args.disqualify_threshold,
            mode=mode_to_set,
            step_callback=on_step,
        )

    if not args.json:
        render_summary_dashboard(
            console,
            result,
            top_tier_threshold=args.top_tier,
            strong_match_threshold=args.strong_match,
            inspect_jobs=getattr(args, "inspect", True),
        )
    else:
        print(json.dumps(result.model_dump(), indent=2, default=str))

    return result


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    try:
        result = asyncio.run(execute_cli_pipeline(args, console=console))
        return 0 if result.success else 1
    except KeyboardInterrupt:
        console.print("\n[bold red]Pipeline execution interrupted by user.[/bold red]")
        return 130
    except Exception as exc:
        console.print(f"\n[bold red]Fatal pipeline error:[/bold red] {exc}")
        if args.verbose:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
