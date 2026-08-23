# Israeli Multi-Source Tech Job FastMCP Server (`TechJobMCP`)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP 2.0+](https://img.shields.io/badge/FastMCP-2.0+-green.svg)](https://github.com/jlowin/fastmcp)
[![Tests Passing](https://img.shields.io/badge/tests-654%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, privacy-first **FastMCP** server providing intelligent, multi-source tech job aggregation, smart deduplication, dynamic CV skill & target role extraction, requirement coverage scoring, zero-guesswork Universal DOM form automation, and autonomous job scouting workflows across **HireMeTech**, **Comeet ATS**, **Workday Enterprise**, **Eightfold.ai**, **DirectTech (Google, Apple, Amazon, IBM)**, and **LinkedIn**.

---

## 📖 Documentation Guides

- 🚀 [**Setup & Candidate Profile Configuration Guide**](docs/SETUP_GUIDE.md) — Dynamic CV extraction, target roles, environment config, Docker & Python local run.
- 🌐 [**Cloudflare Tunnel & Port Export Guide**](docs/TUNNEL_AND_PORTS.md) — Exposing port 8000 via `cloudflared` for remote AI clients.
- 🤖 [**AI Client Integrations Guide**](docs/CLIENT_INTEGRATIONS.md) — Complete setup for **Gemini Spark** (autonomous scheduled scout), **Claude Desktop / CoWork**, **ChatGPT / Codex**, and **Cursor / Antigravity**.
- 🛠️ [**Spark Prompts & Skills**](docs/SPARK_SKILL_PROMPT.md) — System prompt and skill instructions for autonomous scouting agents.

---

## 🏛️ Architecture Overview

```mermaid
graph TD
    Client([MCP Client: Gemini Spark / Claude / Cursor / ChatGPT]) --> Tools[FastMCP Server Layer]
    Tools --> Aggregator[JobAggregator]
    Aggregator --> Registry[SourceRegistry]

    subgraph Parallel Pluggable Sources Layer
        Registry --> S1[HireMeTechSource<br/>Direct REST API + DOM Fallback]
        Registry --> S2[ComeetSource<br/>Direct ATS API + Concurrency Semaphore]
        Registry --> S3[WorkdaySource<br/>Enterprise Workday CXS Direct API]
        Registry --> S4[EightfoldSource<br/>PCSX Search API]
        Registry --> S5[DirectTechSource<br/>Google, Apple, Amazon, IBM Feeds]
        Registry --> S6[LinkedInSource<br/>Job Guest Search API]
    end

    subgraph Processing & Normalization Engine
        S1 --> Dedup[Deduplication & Entity Merger]
        S2 --> Dedup
        S3 --> Dedup
        S4 --> Dedup
        S5 --> Dedup
        S6 --> Dedup
        
        Dedup --> NormKey["Key = slug(title) + '@' + slug(company)"]
        NormKey --> Merge[Metadata & Links Merger]
        Merge --> Scorer[Unified CV / Skill Matcher]
    end

    subgraph Dynamic Candidate Engine
        CV["Candidate CV (.pdf / .docx / .txt)"] --> Extractor[Dynamic CV & Profile Extractor]
        Extractor --> Skills["Extracted Skills (40+ tokens)"]
        Extractor --> Stack["Primary Tech Stack (Top Skills)"]
        Extractor --> Seniority["Inferred Seniority & Exclusions"]
        Extractor --> Roles["Dynamic Target Roles"]
        Skills --> Scorer
        Stack --> Scorer
        Seniority --> Scorer
        Roles --> Scorer
    end

    subgraph Autonomous Application Engine
        Tools --> Dispatcher["HybridApplicationDispatcher"]
        Dispatcher --> Guardrails{"Safety Guardrails<br/>Cap / Dups / Score / IL"}
        Guardrails --> StrategyRouter["Strategy Selector"]
        StrategyRouter --> BrowserStrategy["BrowserPlaywrightStrategy"]
        StrategyRouter --> ApiStrategy["ApiPostStrategy"]
        StrategyRouter --> EasyApply["EasyApplyStrategy"]
        
        BrowserStrategy --> DOMInspector["Universal DOMInspector<br/>Recursive Frames & Zero-Guesswork"]
        BrowserStrategy --> FormMapper["SemanticFormMapper<br/>Regex + Free-Tier LLM Gateway"]
        Dispatcher --> Ledger[("ApplicationLedger (SQLite)")]
    end

    Scorer --> Cache[Unified JobCache - 2h TTL]
    Cache --> Tools
```

---

## 🌟 Key Features

1. **Dynamic Candidate Extraction**:
   - Ingests `.pdf`, `.docx`, and `.txt` resumes for any engineering specialty (Java, Python, Frontend, DevOps, AI, Web3).
   - Automatically derives candidate `primary_stack` and `target_roles` without hardcoded assumptions.
   - Detects seniority level (Student, Junior, Mid, Senior, Lead) and generates tailored negative keywords.

2. **Calibrated Match Scoring (0–100)**:
   - **Primary Stack Affinity**: Heavily weights candidate's core technologies (+35 pts).
   - **Skill Volume Scaling**: Scales with absolute count of matched core competencies (4+ skills = 30 pts; 1 generic skill = 8 pts max).
   - **Target Role Semantic Fit**: Awards +15 pts for developer titles matching target roles; caps non-engineering/administrative positions at 65 pts.
   - **Commute & Location Normalization**: Normalizes country codes (`", IL"`, `Israel`, `ישראל`) and handles peripheral on-site roles.

3. **Universal Dynamic DOM Form Solver**:
   - **Zero-Guesswork DOM Inspection**: In-browser JS evaluation traverses main documents and recursive `iframes` (e.g., Comeet embedded forms, Workday modal overlays).
   - **11-Tier Contextual Label Resolution**: Matches labels using `aria-label`, `<label for>`, preceding text, fieldsets, and placeholders.
   - **Deterministic & AI-Assisted Submission**: Deterministic keyword scoring (+65 to +100 for submit buttons, -100 for negative/cancel actions) with LLM disambiguation fallback.

4. **Resilient Free-Tier LLM Gateway & Caching**:
   - Multi-provider fallback chain (Gemini Flash Lite -> OpenRouter -> Ollama -> Heuristic Mock).
   - Token-bucket rate limiting (15 RPM) and jittered exponential backoff retries.
   - Zero-cost SQLite semantic caching (`llm_cache.db`) for questionnaire answers.

5. **Safety Guardrails & Application Ledger**:
   - Audit log in SQLite (`application_ledger.db`).
   - Duplicate prevention (`is_applied` check).
   - Fail-closed master switch (`AUTO_APPLY_ENABLED=false`).
   - Israel/Remote location enforcement and daily run caps (`MAX_DAILY_APPLICATIONS`).

---

## 🛠️ Tool Reference (15 Tools)

| Tool Name | Parameters | Description |
|---|---|---|
| `run_job_scout` | `cv_path`, `location`, `top_tier_threshold`, `strong_match_threshold`, `disqualify_threshold`, `auto_bookmark`, `auto_apply`, `action_mode`, `force_refresh`, `notify_channel` | **Composite Scout Tool**: Runs end-to-end multi-source aggregation, scoring, bookmarking, and safe application execution in one call. |
| `list_job_sources` | *none* | Lists all registered job sources, capabilities, and real-time health. |
| `get_job_matches` | `sources: list[str] = None`, `force_refresh: bool = False` | Fetches matched listings across all or specified platforms with deduplication. |
| `filter_jobs_by_preferences` | `tech_stack`, `work_mode`, `location`, `min_salary`, `keywords`, `exclude_keywords`, `cv_path` | Scores and filters aggregated jobs against candidate CV and preferences. |
| `bookmark_job` | `job_id: str` | Saves/favorites a job listing on the originating platform. |
| `delete_job` | `job_id: str` | Dismisses/hides a job listing from view and removes it from cache. |
| `auto_apply_job` | `job_id: str`, `cv_path: str = None` | **Step 1**: Inspects application modal, stages dynamic preview, maps form fields, and reports warnings. |
| `confirm_auto_apply` | `job_id: str`, `cv_path: str = None`, `force: bool = False` | **Step 2**: Executes application submission via Playwright DOM / API POST. Requires explicit confirmation or force. |
| `get_application_history` | `limit: int = 50`, `status: str = None` | Retrieves the immutable audit log of past application submissions from `ApplicationLedger`. |
| `calibrate_selectors` | *none* | Discovers and calibrates DOM selectors against live pages with self-healing heuristics. |
| `search_linkedin_jobs` | `keywords: list[str]`, `location: str = "Israel"`, `limit: int = 25` | Dedicated LinkedIn search tool returning normalized Job models. |
| `get_linkedin_job_details` | `job_id: str` | Fetches rich job description and metadata for a specific LinkedIn posting. |
| `notify_new_jobs` | `jobs: list[dict]`, `channel: str = "telegram"` | Sends structured notification digest of new top-tier job opportunities. |
| `test_notifier` | `channel: str = "telegram"` | Tests notification channel configuration. |
| `set_operation_mode` | `mode: 'supervised' \| 'autonomous'` | Switches server execution mode between supervised and autonomous. |

---

## ⚡ Quick Start

### 1. Clone & Configure
```bash
git clone https://github.com/zvieli/TechJobMCP.git
cd TechJobMCP

# Copy your CV and setup environment
cp /path/to/your/resume.pdf ./cv.pdf
cp .env.example .env
```

### 2. Run with Docker Compose
```bash
docker compose up -d
docker compose logs -f techjob-mcp
```

### 3. Export Public HTTPS Port for AI Clients
```bash
curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8000
```

Connect the generated `https://<tunnel-id>.trycloudflare.com/mcp` URL to your AI client. See [**AI Client Integrations Guide**](docs/CLIENT_INTEGRATIONS.md) for full setup instructions.

---

## 🧪 Running Tests

Run the full automated test suite (**654 unit and integration tests**):

```bash
.venv/bin/pytest
```

---

## 📄 License

This project is licensed under the MIT License.
