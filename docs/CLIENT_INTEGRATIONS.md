# AI Client Integrations Guide (Gemini, Claude, ChatGPT, Cursor)

This guide covers connecting **TechJobMCP** to all major LLM agents and AI platforms.

---

## 1. Gemini Spark & Google AI Studio (Autonomous Scheduled Scout)

Gemini Spark supports connecting Custom MCP Apps and running autonomous scheduled job scouting pipelines.

### Step 1: Add the Custom App Connector in Gemini
1. Open Gemini Spark / AI Studio $\rightarrow$ **Settings** $\rightarrow$ **Connected Apps / MCP Connectors**.
2. Click **Add New App**:
   - **Name**: `TechJobMCP`
   - **Endpoint URL**: `https://<your-tunnel-id>.trycloudflare.com/mcp`
   - **Transport**: `Streamable HTTP`
   - **Authentication**: `None`
3. Click **Connect & Verify**. Gemini will discover tools including `run_job_scout`, `get_job_matches`, `filter_jobs_by_preferences`, etc.

---

### Step 2: Define the Autonomous Scouting Skill Prompt

Save the following skill in Gemini (e.g. `/job-opportunity-scout`):

```text
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
separately. Treat the server's scores, actions, blocks, counts, and trace IDs as
authoritative.

Process MCP results in descending score order. Preserve every returned
`trace_id` and show a source badge beside every job. A server-side preview or
submission block is final for this run. Bookmark and report jobs from sources
without MCP submission support with their direct application URLs.

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
```

---

## 2. Claude Desktop & Claude CoWork (Anthropic)

### Option A: Local Stdio Connection (Local Machine)
Add to your Claude Desktop configuration file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "TechJobMCP": {
      "command": "/path/to/TechJobMCP/.venv/bin/python",
      "args": ["-m", "job_mcp", "--transport", "stdio"],
      "env": {
        "DEFAULT_CV_PATH": "/path/to/TechJobMCP/cv.pdf",
        "CANDIDATE_EMAIL": "your.email@example.com",
        "CANDIDATE_NAME": "Your Name",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Option B: Remote HTTP Tunnel Connection (Claude CoWork / Remote Agent)
```json
{
  "mcpServers": {
    "TechJobMCP": {
      "url": "https://<your-tunnel-id>.trycloudflare.com/mcp",
      "transport": "http"
    }
  }
}
```

---

## 3. OpenAI ChatGPT, Custom GPTs & OpenAI Codex

### Connecting as a Custom GPT Action
1. In ChatGPT $\rightarrow$ **Explore GPTs** $\rightarrow$ **Create a GPT** $\rightarrow$ **Configure** $\rightarrow$ **Actions**.
2. Set the Schema URL to your server's OpenAPI definition:
   `https://<your-tunnel-id>.trycloudflare.com/openapi.json`
3. In instructions, prompt ChatGPT:
   ```text
   You are an autonomous Job Search Agent. Always use the TechJobMCP action `run_job_scout` with `cv_path="/app/cv.pdf"`, location="Israel", and rank matches by descending score.
   ```

### Connecting in OpenAI Codex / CLI
If using an MCP-compatible Codex or CLI client, point the client configuration to:
```bash
npx @modelcontextprotocol/inspector https://<your-tunnel-id>.trycloudflare.com/mcp
```

---

## 4. Cursor / Windsurf / Antigravity IDE

In your IDE settings (`~/.cursor/mcp.json` or Antigravity MCP settings):

```json
{
  "mcpServers": {
    "TechJobMCP": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/TechJobMCP", "python", "-m", "job_mcp", "--transport", "stdio"]
    }
  }
}
```

Or via SSE/HTTP tunnel:
```json
{
  "mcpServers": {
    "TechJobMCP": {
      "url": "https://<your-tunnel-id>.trycloudflare.com/mcp"
    }
  }
}
```

---

## 5. Testing Tool Invocations Across Clients

To verify that your connected client can run the full scout pipeline, ask your agent:
> *"Run `run_job_scout` with my CV and summarize top matches in Israel."*

Expected response telemetry:
- **`mcp_status`**: `"success"`
- **`top_tier_jobs`**: List of jobs with match score $\ge 85$.
- **`strong_match_jobs`**: List of jobs with match score $70 - 84$.
- **`bookmarked`**: IDs of saved positions.
- **`summary_text`**: Complete audit report.
