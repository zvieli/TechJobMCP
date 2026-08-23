
Run the /job-opportunity-scout skill now and explicitly use the connected
@TechJobMCP custom app. Control the tools in English, but write the complete
user-facing result in Hebrew.

Call `run_job_scout` exactly once with `action_mode="autonomous"`, sources
`["hiremetech","comeet","workday","eightfold","direct_tech","linkedin"]`,
`/app/cv.pdf`, `location="Israel"`, the configured seniority,
five-or-more-years, foreign-work, and university-only exclusions,
`force_refresh=false`, `detail_level="summary"`, `limit=30`, thresholds
`85/70/50`, and `max_applications=3`. Do not call `set_operation_mode`,
`list_job_sources`, `get_job_matches`, or `filter_jobs_by_preferences`
separately. Treat the server's scores, hybrid application routing (Direct API,
Easy Apply, Playwright Browser), blocks, counts, and trace IDs as authoritative.

Process MCP results in descending score order. Preserve every returned
`trace_id` and show a source badge beside every job. A server-side preview or
submission block (fail-closed guardrails, score threshold, location check,
duplicate ledger check, or daily limit) is final for this run. Bookmark and report
jobs with their direct application URLs if submission is blocked or staged.

Never use Google Search, a browser, another app, or a fallback tool. Never retry
an autonomous scout after a timeout or ambiguous failure unless its structured
response explicitly says `safe_to_retry=true`; actions may already have
occurred. If no structured MCP response was received, report no jobs, preserve
the exact error, use `אין` for trace IDs, and write `לא ידוע` for every action
count. Never report zero for an unknown count. If a structured response says
`coverage_complete=false`, explain that MCP suppressed all actions and list the
returned source statuses.

End with this exact Hebrew audit block:

```text
MCP ראשי: הצליח|נכשל
נעשה שימוש בגיבוי: לא
מזהי מעקב של MCP: <מזהים מופרדים בפסיקים, או "אין">
הוגשו: <מספר מהשרת, או "לא ידוע" אם לא התקבלה תשובת MCP מובנית>
נשמרו כסימנייה: <מספר מהשרת, או "לא ידוע">
הוסרו ממקור תומך: <מספר מהשרת, או "לא ידוע">
הוסרו ממטמון MCP: <מספר מהשרת, או "לא ידוע">
נחסמו: <מספר מהשרת, או "לא ידוע">
נכשלו: <מספר מהשרת, או "לא ידוע">
נדחו לריצה עתידית: <מספר מהשרת, או "לא ידוע">
```

If MCP failed, immediately add:
`כשל MCP: <פעולת MCP>: <השגיאה המדויקת>`.