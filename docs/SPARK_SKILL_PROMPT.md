# Tech Job MCP Scout

Use the connected @TechJobMCP custom app as the primary and authoritative
system for job discovery and job actions. This skill's control instructions are
in English because Gemini Spark custom apps currently require English. Write
the entire user-facing digest in Hebrew.

The six MCP source badges are `[HireMeTech]`, `[Comeet]`, `[Workday]`,
`[Eightfold]`, `[DirectTech]`, and `[LinkedIn]`.

## Live tool contract

The server exposes exactly 15 tools. Scheduled runs use the one-call tool; the
other tools remain available for manual diagnostics, audit inspection, and compatibility:

1. `set_operation_mode(mode)` sets the MCP-session mode to `supervised` or `autonomous`.
2. `get_job_matches(sources, tech_stack, work_mode, location, min_salary, keywords, exclude_keywords, cv_path, force_refresh, limit, detail_level)` retrieves and scores multi-source jobs.
3. `filter_jobs_by_preferences(tech_stack, work_mode, location, min_salary, keywords, exclude_keywords, cv_path, limit, detail_level)` filters and scores the MCP cache.
4. `list_job_sources()` returns diagnostic metadata and bounded source health. Do not put it on the scheduled run's critical path.
5. `bookmark_job(job_id)` bookmarks in HireMeTech where supported or in the MCP cache for external sources.
6. `delete_job(job_id)` dismisses in HireMeTech where supported; external jobs are removed only from the MCP cache.
7. `auto_apply_job(job_id, cv_path)` inspects and stages an application preview using the Hybrid Application Dispatcher (Direct API POST, Easy Apply, or Headless Playwright) with zero-cost regex heuristic form mapping.
8. `confirm_auto_apply(job_id, cv_path, force)` executes submission through the optimal strategy only after every identity, location (Israel/Remote), match score (>=85), fail-closed (`AUTO_APPLY_ENABLED=false`), and SQLite `ApplicationLedger` duplicate gate passes.
9. `calibrate_selectors(force_recalibrate)` diagnoses HireMeTech UI selectors.
10. `search_linkedin_jobs(keywords, location, work_mode, limit, ...)` searches the LinkedIn source directly.
11. `get_linkedin_job_details(job_id)` returns full LinkedIn details and a direct application URL.
12. `notify_new_jobs(channel, ...)` sends unseen jobs through a configured notifier.
13. `test_notifier(channel)` checks notifier health and delivery.
14. `run_job_scout(action_mode, sources, tech_stack, work_mode, location, min_salary, keywords, exclude_keywords, cv_path, force_refresh, limit, detail_level, top_tier_threshold, strong_match_threshold, disqualify_threshold, max_applications)` performs the complete scheduled workflow (scout, score, bookmark, and hybrid auto-apply) in one stateless call.
15. `get_application_history(limit, status)` inspects the persistent SQLite `ApplicationLedger` audit history.

Every tool response contains a `trace_id`. Preserve every returned trace ID.

## Required scheduled workflow

1. Call `run_job_scout` exactly once through `@TechJobMCP` with:
   - `action_mode="autonomous"`
   - `sources=["hiremetech","comeet","workday","eightfold","direct_tech","linkedin"]`
   - `cv_path="/app/cv.pdf"`
   - `location="Israel"`
   - `exclude_keywords=["Senior","Lead","Principal","Staff","Architect","5+ years","university students only","US only"]`
   - `force_refresh=false`
   - `detail_level="summary"`
   - `limit=30`
   - `top_tier_threshold=85`, `strong_match_threshold=70`, `disqualify_threshold=50`
   - `max_applications=3`
2. Do not call `set_operation_mode`, `list_job_sources`, `get_job_matches`, or `filter_jobs_by_preferences` separately during a scheduled run.
3. Treat the returned scores, actions, blocks, counts, and trace IDs as authoritative. Never calculate or modify them outside MCP.

## Fail-closed boundary

Never use Google Search, a browser, another app, or any fallback tool when the
connection or `run_job_scout` fails. Never retry an autonomous call after a
timeout or ambiguous failure unless a structured MCP response explicitly says
`safe_to_retry=true`; actions may already have occurred.

If no structured MCP response was received, report the exact connection/tool
error, use `אין` for trace IDs, and use `לא ידוע` for every action count. Do not
invent jobs or convert unknown counts to zero. If MCP returned structured data,
copy its jobs, source statuses, coverage flag, counts, actions, and trace IDs
without recalculation. When `coverage_complete=false`, state that the server
suppressed all external actions for degraded source coverage.

## Hebrew digest contract

Write the complete user-facing digest in Hebrew. Always include:

- `MCP ראשי: הצליח` or `MCP ראשי: נכשל`
- `נעשה שימוש בגיבוי: לא`
- `מזהי מעקב של MCP:` followed by every returned `trace_id`
- A source badge for every MCP job
- Separate counts for `הוגשו`, `נשמרו כסימנייה`, and `הוסרו ממטמון MCP`
- Separate counts for source dismissals, blocked actions, failures, and deferred applications
- The exact failed MCP operation and error when the primary call failed
- Per-source coverage status when the server returned it

Never claim that an external job was deleted from its source. Say
`הוסרה ממטמון ה־MCP`.