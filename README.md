# HireMeTech FastMCP Server

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP 2.0+](https://img.shields.io/badge/FastMCP-2.0+-green.svg)](https://github.com/jlowin/fastmcp)
[![Playwright](https://img.shields.io/badge/Playwright-Chromium-orange.svg)](https://playwright.dev/python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade **FastMCP** server providing intelligent job matching, filtering, bookmarking, and automated job applications on **HireMeTech** using headless Playwright browser automation and persistent session state.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client                             │
│     (Claude Desktop / Gemini Spark / Antigravity / Cursor)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ JSON-RPC (stdio / HTTP / SSE)
┌──────────────────────────────▼──────────────────────────────┐
│                  HireMeTech FastMCP Server                  │
│                                                             │
│  ┌───────────────────────┐      ┌────────────────────────┐  │
│  │   Lifespan Manager    │      │       Job Cache        │  │
│  │  (Session & Health)   │      │    (TTL In-Memory)     │  │
│  └───────────┬───────────┘      └───────────▲────────────┘  │
│              │                              │               │
│  ┌───────────▼──────────────────────────────┴────────────┐  │
│  │                 6 Core MCP Tools                      │  │
│  │  • get_job_matches        • bookmark_job              │  │
│  │  • filter_jobs_by_prefs   • delete_job                │  │
│  │  • auto_apply_job (Step 1)• confirm_auto_apply(Step 2)│  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              │                              │
│  ┌───────────────────────────▼───────────────────────────┐  │
│  │      Playwright Automation & Self-Healing Engine      │  │
│  │  • Primary / Fallback Selectors Registry              │  │
│  │  • Persistent Chromium Profile Store                  │  │
│  └───────────────────────────┬───────────────────────────┘  │
└──────────────────────────────┼──────────────────────────────┘
                               │ Chromium (Headless/Headed)
┌──────────────────────────────▼──────────────────────────────┐
│                    https://hiremetech.com                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

1. **Persistent Browser Session**: Reusable user profile directory storing login cookies, tokens, and session state across runs.
2. **Self-Healing Selectors**: Primary data-testid selectors with automatic fallbacks for resilient DOM navigation.
3. **Smart Job Matching & Scoring**: Weighted scoring engine (0-100) evaluating tech stack overlap, keywords, work mode, and CV text.
4. **CV / Resume Parser**: Automatically extracts technologies and skill keywords from `.pdf`, `.docx`, and `.txt` files.
5. **Safe 2-Step Application Workflow**:
   - **Step 1 (`auto_apply_job`)**: Non-destructive inspection of form fields and generation of application preview.
   - **Step 2 (`confirm_auto_apply`)**: Explicit user confirmation before submission.
6. **Multi-Transport Support**: Run over `stdio`, `http`, or `sse`.

---

## Tool Reference

| Tool Name | Parameters | Description |
|---|---|---|
| `get_job_matches` | `force_refresh: bool = False` | Fetches jobs from HireMeTech dashboard. Uses cache if fresh. |
| `filter_jobs_by_preferences` | `tech_stack: list[str]`, `work_mode: str`, `location: str`, `min_salary: int`, `keywords: list[str]`, `exclude_keywords: list[str]`, `cv_path: str` | Filters and ranks cached jobs with match scoring (0-100). |
| `bookmark_job` | `job_id: str` | Bookmarks/favorites a job listing on the platform and in cache. |
| `delete_job` | `job_id: str` | Dismisses/hides a job listing from view and removes from cache. |
| `auto_apply_job` | `job_id: str` | **Step 1**: Inspects application modal, stages preview, reports warnings. |
| `confirm_auto_apply` | `job_id: str` | **Step 2**: Executes application submission for a previously staged preview. |

---

## Installation & Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-username/hireme_mcp.git
cd hireme_mcp

# Using uv (recommended)
uv venv .venv
uv pip install -e ".[dev]"

# Install Playwright browser dependencies
playwright install chromium
```

### 2. First-Time Authentication Setup

Run the interactive setup utility to open a headed Chromium window and log in to your HireMeTech account:

```bash
# Using CLI entry point
python -m hireme_mcp.setup

# Or via console script
hireme-mcp-setup
```

1. Log in to your HireMeTech account in the opened browser window (complete any 2FA/SSO if prompted).
2. Once on the Dashboard, return to the terminal and press `[Enter]`.
3. Your session cookies and tokens are securely saved to `~/.hireme_mcp/browser_profile`.

---

## Running the Server

### Stdio Mode (Default for MCP Clients)

```bash
python -m hireme_mcp
```

### HTTP Transport Mode

```bash
export MCP_TRANSPORT=http
export MCP_HOST=0.0.0.0
export MCP_PORT=8000
python -m hireme_mcp
```

### SSE Transport Mode

```bash
export MCP_TRANSPORT=sse
export MCP_HOST=0.0.0.0
export MCP_PORT=8000
python -m hireme_mcp
```

---

## MCP Client Configuration

### Claude Desktop (`claude_desktop_config.json`)

**On Linux/macOS:** `~/.config/Claude/claude_desktop_config.json`  
**On Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hireme-tech": {
      "command": "/home/lior/data/projects/hireme_mcp/.venv/bin/python",
      "args": ["-m", "hireme_mcp"],
      "env": {
        "BROWSER_HEADLESS": "true",
        "BROWSER_PROFILE_DIR": "/home/lior/.hireme_mcp/browser_profile",
        "CACHE_TTL_MINUTES": "60",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Gemini Spark / Antigravity / Cursor (`mcp_config.json`)

```json
{
  "mcpServers": {
    "hireme-tech": {
      "command": "python",
      "args": ["-m", "hireme_mcp"],
      "transport": "stdio",
      "env": {
        "BROWSER_HEADLESS": "true",
        "BROWSER_PROFILE_DIR": "~/.hireme_mcp/browser_profile"
      }
    }
  }
}
```

---

## Docker Deployment

### Using Docker Compose

```bash
# 1. Build and run
docker compose up -d

# 2. View logs
docker compose logs -f hireme-mcp
```

### Using Docker Directly

```bash
# Build image
docker build -t hireme-mcp .

# Run container with mapped persistent profile
docker run -d \
  --name hireme-mcp-server \
  -p 8000:8000 \
  -v $(pwd)/browser_profile:/app/browser_profile \
  -e MCP_TRANSPORT=http \
  -e BROWSER_HEADLESS=true \
  hireme-mcp
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | Transport protocol (`stdio`, `http`, `sse`). |
| `MCP_HOST` | `0.0.0.0` | Host binding for HTTP/SSE transport. |
| `MCP_PORT` | `8000` | Port for HTTP/SSE transport. |
| `BROWSER_HEADLESS` | `true` | Run browser in headless mode (`true`/`false`). |
| `BROWSER_PROFILE_DIR` | `~/.hireme_mcp/browser_profile` | Directory for persistent Chromium storage. |
| `CACHE_TTL_MINUTES` | `60` | In-memory job cache expiration time in minutes. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Running Tests

Run the full automated test suite with pytest:

```bash
.venv/bin/pytest tests/ -v
```

Output:
```
tests/test_core.py::TestAuth::test_auth_constants PASSED
tests/test_core.py::TestAuth::test_session_manager_custom_init PASSED
tests/test_core.py::TestAuth::test_session_manager_init_defaults PASSED
tests/test_core.py::TestAuth::test_session_manager_lifecycle PASSED
tests/test_core.py::TestBrowser::test_bookmark_and_delete_job PASSED
tests/test_core.py::TestBrowser::test_extract_jobs_mock PASSED
tests/test_core.py::TestBrowser::test_helper_extractors PASSED
tests/test_core.py::TestBrowser::test_preview_and_execute_application PASSED
tests/test_core.py::TestBrowser::test_resolve_selector_primary PASSED
tests/test_core.py::TestBrowser::test_selectors_registry PASSED
tests/test_core.py::TestApiClient::test_extract_cv_keywords_docx PASSED
tests/test_core.py::TestApiClient::test_extract_cv_keywords_txt PASSED
tests/test_core.py::TestApiClient::test_filter_jobs PASSED
tests/test_core.py::TestApiClient::test_job_cache PASSED
tests/test_server.py::TestServerRegistration::test_all_tools_registered PASSED
tests/test_server.py::TestServerRegistration::test_browser_lifespan PASSED
tests/test_server.py::TestServerRegistration::test_server_metadata PASSED
tests/test_server.py::TestCliAndSetup::test_main_http PASSED
tests/test_server.py::TestCliAndSetup::test_main_sse PASSED
tests/test_server.py::TestCliAndSetup::test_main_stdio PASSED
tests/test_server.py::TestCliAndSetup::test_run_setup_failure PASSED
tests/test_server.py::TestCliAndSetup::test_run_setup_success PASSED
tests/test_tools.py::TestMcpTools::test_bookmark_job_flow PASSED
tests/test_tools.py::TestMcpTools::test_confirm_auto_apply_without_preview_error PASSED
tests/test_tools.py::TestMcpTools::test_delete_job_flow PASSED
tests/test_tools.py::TestMcpTools::test_filter_jobs_by_cv_file PASSED
tests/test_tools.py::TestMcpTools::test_filter_jobs_by_stack_and_work_mode PASSED
tests/test_tools.py::TestMcpTools::test_filter_jobs_no_cached_jobs PASSED
tests/test_tools.py::TestMcpTools::test_get_job_matches_cache_hit PASSED
tests/test_tools.py::TestMcpTools::test_get_job_matches_live_fetch_and_force_refresh PASSED
tests/test_tools.py::TestMcpTools::test_get_job_matches_unauthenticated PASSED
tests/test_tools.py::TestMcpTools::test_two_step_auto_apply_flow PASSED
```

---

## License

This project is licensed under the MIT License.
