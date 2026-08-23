"""Mock LLM Agent and execution engine for Tech Job  MCP testing."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Optional

from fastmcp import Context
from pydantic import BaseModel, Field

from job_mcp.core.api_client import extract_candidate_profile, resolve_cv_path
from job_mcp.main import (
    auto_apply_job,
    bookmark_job,
    calibrate_selectors,
    confirm_auto_apply,
    delete_job,
    filter_jobs_by_preferences,
    get_job_matches,
    list_job_sources,
    run_job_scout,
    set_operation_mode,
)
from job_mcp.models.schemas import CandidateProfile


class StepTrace(BaseModel):
    """Execution step trace recorded during tool invocation or reasoning."""

    step_number: int
    thought: str = ""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class PipelineResult(BaseModel):
    """Structured outcome of an end-to-end LLM job hunt pipeline execution."""

    success: bool
    profile: Optional[CandidateProfile] = None
    steps: list[StepTrace] = Field(default_factory=list)
    sources_found: list[str] = Field(default_factory=list)
    total_jobs_fetched: int = 0
    top_tier_jobs: list[dict[str, Any]] = Field(default_factory=list)
    strong_match_jobs: list[dict[str, Any]] = Field(default_factory=list)
    bookmarked_job_ids: list[str] = Field(default_factory=list)
    staged_apply_ids: list[str] = Field(default_factory=list)
    confirmed_apply_ids: list[str] = Field(default_factory=list)
    deleted_job_ids: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0


TOOL_DISPATCH: dict[str, Callable[..., Any]] = {
    "list_job_sources": list_job_sources,
    "get_job_matches": get_job_matches,
    "filter_jobs_by_preferences": filter_jobs_by_preferences,
    "bookmark_job": bookmark_job,
    "delete_job": delete_job,
    "auto_apply_job": auto_apply_job,
    "confirm_auto_apply": confirm_auto_apply,
    "set_operation_mode": set_operation_mode,
    "calibrate_selectors": calibrate_selectors,
    "run_job_scout": run_job_scout,
}


class MockLLMAgent:
    """Mock LLM Agent simulating autonomous job search, matching, and application."""

    def __init__(
        self,
        mcp_server: Optional[Any] = None,
        cv_path: Optional[str] = None,
        context: Optional[Context] = None,
    ) -> None:
        """Initialize the MockLLMAgent.

        Args:
            mcp_server: Optional FastMCP server instance or client reference.
            cv_path: Optional path to candidate's CV file.
            context: Optional FastMCP Context object for tool execution.
        """
        self.mcp_server = mcp_server
        self.cv_path = cv_path
        self.context = context
        self.history: list[StepTrace] = []

    def reset_history(self) -> None:
        """Clear recorded step history."""
        self.history.clear()

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        thought: Optional[str] = None,
        step_callback: Optional[Callable[[StepTrace], Any]] = None,
    ) -> dict[str, Any]:
        """Execute a tool function by name, measure duration, and log a StepTrace.

        Args:
            tool_name: Name of tool to execute.
            arguments: Dictionary of arguments to pass to the tool.
            thought: Optional reasoning thought string for this step.
            step_callback: Optional callback invoked with the recorded StepTrace.

        Returns:
            dict[str, Any]: Response dictionary returned by the tool.
        """
        args = dict(arguments or {})
        start_time = time.perf_counter()

        if self.mcp_server is not None and hasattr(self.mcp_server, "call_tool"):
            try:
                raw_res = await self.mcp_server.call_tool(name=tool_name, arguments=args)
                if hasattr(raw_res, "model_dump"):
                    res = raw_res.model_dump()
                elif hasattr(raw_res, "data"):
                    res = raw_res.data if isinstance(raw_res.data, dict) else {"data": raw_res.data, "success": True}
                elif isinstance(raw_res, dict):
                    res = raw_res
                elif isinstance(raw_res, list) and len(raw_res) > 0 and hasattr(raw_res[0], "text"):
                    import json
                    try:
                        res = json.loads(raw_res[0].text)
                    except Exception:
                        res = {"data": raw_res[0].text, "success": True}
                else:
                    res = {"data": raw_res, "success": True}
            except Exception as exc:
                res = {
                    "success": False,
                    "message": f"Error executing tool '{tool_name}' on remote MCP: {exc}",
                    "error_code": "REMOTE_TOOL_EXECUTION_ERROR",
                }
        else:
            func = TOOL_DISPATCH.get(tool_name)
            if func is None:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                error_response = {
                    "success": False,
                    "message": f"Unknown tool: '{tool_name}'",
                    "error_code": "UNKNOWN_TOOL",
                }
                trace = StepTrace(
                    step_number=len(self.history) + 1,
                    thought=thought or f"Calling tool {tool_name}",
                    tool_name=tool_name,
                    arguments=args,
                    response=error_response,
                    duration_ms=round(duration_ms, 2),
                )
                self.history.append(trace)
                if step_callback:
                    try:
                        cb_res = step_callback(trace)
                        if asyncio.iscoroutine(cb_res):
                            await cb_res
                    except Exception:
                        pass
                return error_response

            # Pass context if provided and not explicitly set in args
            call_kwargs = dict(args)
            if "ctx" not in call_kwargs and self.context is not None:
                call_kwargs["ctx"] = self.context

            try:
                raw_res = func(**call_kwargs)
                if asyncio.iscoroutine(raw_res):
                    res = await raw_res
                else:
                    res = raw_res

                if hasattr(res, "model_dump"):
                    res = res.model_dump()
                elif not isinstance(res, dict):
                    res = {"data": res, "success": True}
            except Exception as exc:
                res = {
                    "success": False,
                    "message": f"Error executing tool '{tool_name}': {exc}",
                    "error_code": "TOOL_EXECUTION_ERROR",
                }

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        trace = StepTrace(
            step_number=len(self.history) + 1,
            thought=thought or f"Calling tool {tool_name}",
            tool_name=tool_name,
            arguments=args,
            response=res,
            duration_ms=round(duration_ms, 2),
        )
        self.history.append(trace)
        if step_callback:
            try:
                cb_res = step_callback(trace)
                if asyncio.iscoroutine(cb_res):
                    await cb_res
            except Exception:
                pass
        return res

    async def run_pipeline(
        self,
        tech_stack: Optional[list[str]] = None,
        exclude_keywords: Optional[list[str]] = None,
        cv_path: Optional[str] = None,
        top_tier_threshold: int = 85,
        strong_match_threshold: int = 70,
        auto_apply: bool = True,
        target_roles: Optional[list[str]] = None,
        work_mode: Optional[str] = None,
        location: Optional[str] = None,
        min_salary: Optional[int] = None,
        keywords: Optional[list[str]] = None,
        force_refresh: bool = False,
        disqualify_threshold: int = 50,
        mode: Optional[str] = None,
        step_callback: Optional[Callable[[StepTrace], Any]] = None,
    ) -> PipelineResult:
        """Run an end-to-end autonomous job hunting pipeline.

        Pipeline workflow:
        1. Set operation mode if requested (`set_operation_mode`).
        2. Discover available job platforms via `list_job_sources`.
        3. Fetch aggregated job listings across platforms via `get_job_matches`.
        4. Score and rank jobs against user preferences and CV via `filter_jobs_by_preferences`.
        5. Bookmark and auto-apply (preview + confirm) to Top-Tier jobs (score >= top_tier_threshold).
        6. Bookmark Strong Match jobs (strong_match_threshold <= score < top_tier_threshold).
        7. Delete/dismiss disqualified jobs (score < disqualify_threshold or excluded).

        Args:
            tech_stack: Target technologies (e.g. ['Python', 'FastAPI']).
            exclude_keywords: Keywords to exclude/filter out.
            cv_path: Path to CV file (overrides self.cv_path if specified).
            top_tier_threshold: Minimum match score for top-tier jobs (default: 85).
            strong_match_threshold: Minimum match score for strong match jobs (default: 70).
            auto_apply: Whether to automatically preview and confirm top-tier applications.
            target_roles: Target job roles or titles to search for.
            work_mode: Work mode filter ('remote', 'hybrid', 'onsite').
            location: Location filter string.
            min_salary: Minimum desired annual salary.
            keywords: Additional keywords to match.
            force_refresh: Force refresh scraping from live sources.
            disqualify_threshold: Score below which jobs are considered disqualified (default: 50).
            mode: Optional operation mode ('autonomous' or 'supervised').
            step_callback: Optional callback invoked with each StepTrace.

        Returns:
            PipelineResult: Comprehensive execution results and traces.
        """
        pipeline_start_step_idx = len(self.history)
        pipeline_start_time = time.perf_counter()

        # Step 0: Set operation mode if requested
        if mode:
            await self.call_tool(
                "set_operation_mode",
                arguments={"mode": mode},
                thought=f"Setting operation mode to '{mode}'...",
                step_callback=step_callback,
            )

        effective_cv = cv_path or self.cv_path or os.getenv("DEFAULT_CV_PATH")
        profile: Optional[CandidateProfile] = None
        if effective_cv:
            resolved_cv = resolve_cv_path(effective_cv)
            if resolved_cv:
                effective_cv = str(resolved_cv)
            profile = extract_candidate_profile(effective_cv)

        # Dynamic population from CandidateProfile if not explicitly passed
        effective_tech_stack = list(tech_stack) if tech_stack else None
        if effective_tech_stack is None and profile is not None:
            stack_source = profile.primary_stack or profile.top_skills or profile.skills
            if stack_source:
                effective_tech_stack = list(stack_source)

        effective_exclude = list(exclude_keywords) if exclude_keywords is not None else None
        if effective_exclude is None and profile is not None and profile.suggested_exclusions:
            effective_exclude = list(profile.suggested_exclusions)

        effective_keywords = list(keywords) if keywords is not None else None
        if effective_keywords is None and profile is not None and profile.search_queries:
            effective_keywords = list(profile.search_queries)

        effective_target_roles = list(target_roles) if target_roles is not None else None
        if effective_target_roles is None and profile is not None and profile.target_roles:
            effective_target_roles = list(profile.target_roles)

        combined_keywords = list(effective_keywords or [])
        if effective_target_roles:
            for r in effective_target_roles:
                if r not in combined_keywords:
                    combined_keywords.append(r)

        sources_found: list[str] = []
        total_jobs_fetched: int = 0
        top_tier_jobs: list[dict[str, Any]] = []
        strong_match_jobs: list[dict[str, Any]] = []
        bookmarked_job_ids: list[str] = []
        staged_apply_ids: list[str] = []
        confirmed_apply_ids: list[str] = []
        deleted_job_ids: list[str] = []

        # Step 1: Discover available platforms
        res1 = await self.call_tool(
            "list_job_sources",
            arguments={},
            thought="Discovering available job platforms...",
            step_callback=step_callback,
        )
        if res1.get("success"):
            data1 = res1.get("data")
            if isinstance(data1, dict):
                for s in data1.get("sources", []):
                    if isinstance(s, dict):
                        sid = s.get("source_id") or s.get("name")
                        if sid:
                            sources_found.append(sid)
                    elif isinstance(s, str):
                        sources_found.append(s)

        # Step 2: Fetch aggregated jobs
        get_matches_args: dict[str, Any] = {"force_refresh": force_refresh}
        if effective_cv is not None:
            get_matches_args["cv_path"] = effective_cv
        if effective_tech_stack is not None:
            get_matches_args["tech_stack"] = effective_tech_stack
        if effective_exclude is not None:
            get_matches_args["exclude_keywords"] = effective_exclude
        if combined_keywords:
            get_matches_args["keywords"] = combined_keywords
        if work_mode is not None:
            get_matches_args["work_mode"] = work_mode
        if location is not None:
            get_matches_args["location"] = location
        if min_salary is not None:
            get_matches_args["min_salary"] = min_salary

        res2 = await self.call_tool(
            "get_job_matches",
            arguments=get_matches_args,
            thought="Fetching aggregated jobs across sources...",
            step_callback=step_callback,
        )
        raw_jobs: list[dict[str, Any]] = []
        if res2.get("success"):
            data2 = res2.get("data", [])
            if isinstance(data2, list):
                raw_jobs = data2
                total_jobs_fetched = len(raw_jobs)

        # Step 3: Scoring jobs against preferences and CV
        filter_args: dict[str, Any] = {}
        if effective_tech_stack is not None:
            filter_args["tech_stack"] = effective_tech_stack
        if effective_exclude is not None:
            filter_args["exclude_keywords"] = effective_exclude
        if effective_cv is not None:
            filter_args["cv_path"] = effective_cv
        if work_mode is not None:
            filter_args["work_mode"] = work_mode
        if location is not None:
            filter_args["location"] = location
        if min_salary is not None:
            filter_args["min_salary"] = min_salary
        if combined_keywords:
            filter_args["keywords"] = combined_keywords

        res3 = await self.call_tool(
            "filter_jobs_by_preferences",
            arguments=filter_args,
            thought="Scoring jobs against preferences and CV...",
            step_callback=step_callback,
        )
        scored_jobs: list[dict[str, Any]] = []
        if res3.get("success"):
            data3 = res3.get("data", [])
            if isinstance(data3, list):
                scored_jobs = data3

        scored_job_ids = set()
        for job in scored_jobs:
            jid = job.get("job_id")
            if jid:
                scored_job_ids.add(jid)
            score = job.get("match_score")
            score_val = float(score) if score is not None else 0.0

            if score_val >= top_tier_threshold:
                top_tier_jobs.append(job)
            elif score_val >= strong_match_threshold:
                strong_match_jobs.append(job)

        # Disqualified jobs: scored jobs with score < disqualify_threshold, plus raw jobs filtered out entirely
        disqualified_jobs: list[dict[str, Any]] = []
        seen_disqualified = set()

        for job in scored_jobs:
            score = job.get("match_score")
            if score is not None and float(score) < disqualify_threshold:
                jid = job.get("job_id")
                if jid and jid not in seen_disqualified:
                    disqualified_jobs.append(job)
                    seen_disqualified.add(jid)

        for job in raw_jobs:
            jid = job.get("job_id")
            if jid and jid not in scored_job_ids and jid not in seen_disqualified:
                disqualified_jobs.append(job)
                seen_disqualified.add(jid)

        # Step 4: Processing Top-Tier jobs (score >= top_tier_threshold)
        for job in top_tier_jobs:
            jid = job.get("job_id")
            if not jid:
                continue
            title = job.get("title", "Unknown")
            company = job.get("company", "Unknown")
            b_res = await self.call_tool(
                "bookmark_job",
                arguments={"job_id": jid},
                thought=f"Processing Top-Tier jobs (score >= {top_tier_threshold}): bookmarking '{jid}' ({title} at {company})...",
                step_callback=step_callback,
            )
            if b_res.get("success"):
                bookmarked_job_ids.append(jid)

            if auto_apply:
                apply_args: dict[str, Any] = {"job_id": jid}
                if effective_cv:
                    apply_args["cv_path"] = effective_cv
                a_res = await self.call_tool(
                    "auto_apply_job",
                    arguments=apply_args,
                    thought=f"Generating application preview for Top-Tier job '{jid}'...",
                    step_callback=step_callback,
                )
                if a_res.get("success"):
                    staged_apply_ids.append(jid)
                    confirm_args: dict[str, Any] = {"job_id": jid, "force": True}
                    if effective_cv:
                        confirm_args["cv_path"] = effective_cv
                    c_res = await self.call_tool(
                        "confirm_auto_apply",
                        arguments=confirm_args,
                        thought=f"Confirming auto-apply for Top-Tier job '{jid}'...",
                        step_callback=step_callback,
                    )
                    if c_res.get("success"):
                        confirmed_apply_ids.append(jid)

        # Step 5: Processing Strong Match jobs (strong_match_threshold <= score < top_tier_threshold)
        for job in strong_match_jobs:
            jid = job.get("job_id")
            if not jid:
                continue
            title = job.get("title", "Unknown")
            company = job.get("company", "Unknown")
            b_res = await self.call_tool(
                "bookmark_job",
                arguments={"job_id": jid},
                thought=f"Processing Strong Match jobs (score {strong_match_threshold}-{top_tier_threshold - 1}): bookmarking '{jid}' ({title} at {company})...",
                step_callback=step_callback,
            )
            if b_res.get("success"):
                bookmarked_job_ids.append(jid)

        # Step 6: Cleaning up disqualified jobs (score < disqualify_threshold or excluded)
        for job in disqualified_jobs:
            jid = job.get("job_id")
            if not jid:
                continue
            d_res = await self.call_tool(
                "delete_job",
                arguments={"job_id": jid},
                thought=f"Cleaning up disqualified jobs (score < {disqualify_threshold} or excluded): deleting '{jid}'...",
                step_callback=step_callback,
            )
            if d_res.get("success"):
                deleted_job_ids.append(jid)

        execution_time_ms = round((time.perf_counter() - pipeline_start_time) * 1000.0, 2)
        pipeline_steps = self.history[pipeline_start_step_idx:]
        pipeline_success = (
            res1.get("success", False)
            and res2.get("success", False)
            and res3.get("success", False)
        )

        return PipelineResult(
            success=pipeline_success,
            profile=profile,
            steps=pipeline_steps,
            sources_found=sources_found,
            total_jobs_fetched=total_jobs_fetched,
            top_tier_jobs=top_tier_jobs,
            strong_match_jobs=strong_match_jobs,
            bookmarked_job_ids=bookmarked_job_ids,
            staged_apply_ids=staged_apply_ids,
            confirmed_apply_ids=confirmed_apply_ids,
            deleted_job_ids=deleted_job_ids,
            execution_time_ms=execution_time_ms,
        )
