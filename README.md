# Israeli Multi-Source Tech Job FastMCP Server (`TechJobMCP`)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP 2.0+](https://img.shields.io/badge/FastMCP-2.0+-green.svg)](https://github.com/jlowin/fastmcp)
[![Tests Passing](https://img.shields.io/badge/tests-948%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, privacy-first **FastMCP** server providing intelligent, multi-source tech job aggregation, smart deduplication, dynamic CV skill & target role extraction, requirement coverage scoring, zero-guesswork Universal DOM form automation, and autonomous job scouting workflows across **11 Israeli & Global sources**: **Comeet ATS**, **Greenhouse**, **Lever**, **Workday Enterprise**, **Eightfold.ai**, **DirectTech (Google, Apple, Amazon, IBM)**, **LinkedIn (Easy Apply & Guest Search)**, **HireMeTech**, **AllJobs**, **GotFriends**, and **Jobify**.

---

## 📖 Documentation Guides

- 🚀 [**Setup & Candidate Profile Configuration Guide**](docs/SETUP_GUIDE.md) — Dynamic CV extraction, System 1 LAYA setup, System 2 auto-tailoring, Docker & Python local run.
- 🌐 [**Cloudflare Tunnel & Port Export Guide**](docs/TUNNEL_AND_PORTS.md) — Exposing port 8000 via `cloudflared` for remote AI clients.
- 🤖 [**AI Client Integrations Guide**](docs/CLIENT_INTEGRATIONS.md) — Complete setup for **Gemini Spark** (autonomous scheduled scout with receipts), **Claude Desktop / CoWork**, **ChatGPT / Codex**, and **Cursor / Antigravity**.
- 🛠️ [**Spark Prompts & Skills**](docs/SPARK_SKILL_PROMPT.md) — System prompt and skill instructions for autonomous scouting agents with 11 source badges and audit receipts.

---

## 🏛️ Architecture Overview

```mermaid
graph TD
    Client([MCP Client: Gemini Spark / Claude / Cursor / ChatGPT]) --> Tools[FastMCP Server Layer (16 Tools)]
    Tools --> Aggregator[JobAggregator]
    Aggregator --> Registry[SourceRegistry]

    subgraph Parallel Pluggable Sources Layer (11 Sources)
        Registry --> S1[HireMeTechSource<br/>Direct REST API + DOM Fallback]
        Registry --> S2[LinkedInSource<br/>Guest Search + Easy Apply Engine]
        Registry --> S3[ComeetSource<br/>Direct ATS API + iFrame Automation]
        Registry --> S4[GreenhouseSource<br/>Public Boards API + DOM Solver]
        Registry --> S5[LeverSource<br/>Direct API + Structured Postings]
        Registry --> S6[WorkdaySource<br/>Enterprise Workday CXS Direct API]
        Registry --> S7[EightfoldSource<br/>PCSX Search API]
        Registry --> S8[DirectTechSource<br/>Google, Apple, Amazon, IBM Feeds]
        Registry --> S9[JobifySource<br/>Israel Regional Tech Aggregator]
        Registry --> S10[GotFriendsSource<br/>Direct Agency Feed]
        Registry --> S11[AllJobsSource<br/>Portal API & Feed]
    end

    subgraph Processing & Normalization Engine
        S1 & S2 & S3 & S4 & S5 & S6 & S7 & S8 & S9 & S10 & S11 --> Dedup[Deduplication & Entity Merger]
        Dedup --> NormKey["Key = slug(title) + '@' + slug(company)"]
        NormKey --> Merge[Metadata & Links Merger]
    end

    subgraph Dynamic Candidate Engine
        CV["Candidate CV (.pdf / .docx / .txt)"] --> Extractor[Dynamic CV & Profile Extractor]
        Extractor --> Skills["Extracted Skills (40+ tokens)"]
        Extractor --> Stack["Primary Tech Stack (Top Skills)"]
        Extractor --> Seniority["Inferred Seniority & Exclusions"]
        Extractor --> Roles["Dynamic Target Roles"]
    end

    subgraph System 1: Neural Triage Engine (Sub-Millisecond)
        Merge --> Triage{"System 1 Gate"}
        Extractor --> Triage
        Triage --> LayaFT["Fine-Tuned LAYA Engine<br/>Llama-3.2-3B (<1ms Inference)"]
        Triage --> LayaBase["Base LAYA Engine<br/>Zero-Shot Evaluation"]
        Triage --> Heuristic["Deterministic Fallback<br/>Rule-Based Scoring"]
        LayaFT & LayaBase & Heuristic --> SeniorityCap["Seniority & Location Filter<br/>Disqualify or Score (0-100)"]
    end

    subgraph System 2: Cognitive Tailoring Engine
        SeniorityCap --> HighMatch{"Strong Match?<br/>Score ≥ 70/85"}
        HighMatch -- Yes --> Sys2["ApplicationTailoringEngine"]
        Sys2 --> TailoredCV["Custom CV Highlights"]
        Sys2 --> CoverLetter["Tailored Cover Letter (EN/HE)"]
        Sys2 --> Pitch["Recruiter Pitch"]
        Sys2 --> InterviewPrep["Anticipated Interview Questions"]
    end

    subgraph Autonomous Application Engine
        HighMatch --> Dispatcher["HybridApplicationDispatcher"]
        TailoredCV & CoverLetter --> Dispatcher
        Dispatcher --> Guardrails{"Safety Guardrails<br/>Score ≥ 85 / Israel Only / Daily Cap"}
        Guardrails --> StrategyRouter["Strategy Selector"]
        
        StrategyRouter --> EasyApply["EasyApplyStrategy<br/>Multi-Step Traversal (Up to 8 Steps)"]
        StrategyRouter --> BrowserStrategy["BrowserPlaywrightStrategy<br/>Comeet iFrame, Greenhouse, Lever, Workday"]
        StrategyRouter --> ApiStrategy["ApiPostStrategy<br/>Direct ATS JSON Dispatch"]

        EasyApply & BrowserStrategy & ApiStrategy --> Receipts{"4-Layer Verification Receipts"}
        Receipts --> L1["Layer 1: DOM Success Text"]
        Receipts --> L2["Layer 2: Redirect URL"]
        Receipts --> L3["Layer 3: Network HTTP 200/201"]
        Receipts --> L4["Layer 4: Timestamped Screenshot"]

        Receipts --> Ledger[("ApplicationLedger (SQLite)<br/>receipt_details JSON Proof")]
    end

    SeniorityCap --> Cache[Unified JobCache - 2h TTL]
    Cache --> Tools
```

---

## 🌟 Key Features

1. **11 Parallel Pluggable Sources**:
   - Comprehensive multi-source aggregation across **Comeet**, **Greenhouse**, **Lever**, **Workday**, **Eightfold.ai**, **DirectTech (Google, Apple, Amazon, IBM)**, **LinkedIn**, **HireMeTech**, **AllJobs**, **GotFriends**, and **Jobify**.
   - Resilient concurrency with semaphore throttling, rate limiting, and cross-platform deduplication.

2. **System 1 Neural Triage Engine (<1ms Inference)**:
   - **Local AI Youth Assistant (LAYA)**: Specialized fine-tuned model (based on Llama-3.2-3B) performing sub-millisecond candidate-job semantic alignment.
   - **Seniority & Mismatch Capping**: Immediately gates student/junior profiles from senior/lead positions, eliminating wasteful downstream processing.
   - **Pluggable Architecture**: Dependency-injected scoring engine supports `laya` (fine-tuned), `base_laya` (zero-shot baseline), `generative` (LLM gateway), or `heuristic` rule-based scoring.

3. **System 2 Cognitive Tailoring**:
   - Generates bespoke, job-specific application artifacts grounded with Pydantic models:
     - **Tailored CV Highlights**: Specific bullet points mapped to job requirements.
     - **Custom Cover Letters**: Professional, bilingual (English/Hebrew) cover letters matching company culture.
     - **Recruiter Outreach Pitches**: Compelling elevator pitches for recruiters and hiring managers.
     - **Interview Preparation**: Anticipated technical and behavioral questions based on candidate gaps and job specs.

4. **Universal Auto-Apply & 4-Layer Verification Receipts**:
   - **Multi-Step LinkedIn Easy Apply**: Navigates multi-step forms (up to 8 steps) with dynamic semantic form answering.
   - **Universal Browser Solver**: Handles nested `iframe` architectures (e.g. Comeet embedded widgets), modal overlays, and dynamic ATS flows.
   - **4-Layer Verification Receipts**:
     - *Layer 1 (DOM)*: Confirmation text matching ("Application submitted", "Thank you").
     - *Layer 2 (URL)*: Redirect validation to confirmation paths (`/thank-you`, `/submitted`).
     - *Layer 3 (Network)*: Intercepts HTTP 200/201 responses on submission endpoints.
     - *Layer 4 (Screenshot Proof)*: Captures timestamped visual proof to disk (`./data/receipts/*.png`).

5. **Dynamic Candidate Profiling**:
   - Ingests `.pdf`, `.docx`, and `.txt` resumes for any technical discipline.
   - Derives technical skills, primary stack, and dynamic target roles without hardcoded templates.

6. **Safety Guardrails & Application Ledger**:
   - Immutable audit trail in SQLite (`application_ledger.db`) storing full submission receipts.
   - Strict fail-closed master switch (`AUTO_APPLY_ENABLED=false`).
   - Location enforcement (Israel/Remote only) and daily submission quotas (`MAX_DAILY_APPLICATIONS`).

---

## 🛠️ Tool Reference (16 Tools)

| Tool Name | Parameters | Description |
|---|---|---|
| `run_job_scout` | `cv_path`, `location`, `top_tier_threshold`, `strong_match_threshold`, `disqualify_threshold`, `auto_bookmark`, `auto_apply`, `action_mode`, `force_refresh`, `notify_channel`, `max_applications` | **Composite Scout Tool**: Runs end-to-end multi-source aggregation, System 1 scoring, System 2 tailoring, bookmarking, and safe application execution in one call. |
| `list_job_sources` | *none* | Lists all 11 registered job sources, capabilities, and real-time health. |
| `get_job_matches` | `sources: list[str] = None`, `force_refresh: bool = False` | Fetches matched listings across all or specified platforms with deduplication. |
| `filter_jobs_by_preferences` | `tech_stack`, `work_mode`, `location`, `min_salary`, `keywords`, `exclude_keywords`, `cv_path` | Scores and filters aggregated jobs against candidate CV and preferences using System 1. |
| `bookmark_job` | `job_id: str` | Saves/favorites a job listing on the originating platform or local cache. |
| `delete_job` | `job_id: str` | Dismisses/hides a job listing from view and removes it from cache. |
| `auto_apply_job` | `job_id: str`, `cv_path: str = None` | **Step 1**: Inspects application modal, stages dynamic preview, maps form fields, and reports warnings. |
| `confirm_auto_apply` | `job_id: str`, `cv_path: str = None`, `force: bool = False` | **Step 2**: Executes application submission via Easy Apply, Playwright DOM, or API POST. Captures 4-layer receipts. |
| `get_application_history` | `limit: int = 50`, `status: str = None` | Retrieves the immutable audit log and submission receipts from `ApplicationLedger`. |
| `mark_job_as_applied` | `job_id: str`, `source: str`, `company: str`, `title: str`, `notes: str = None` | Manually records an application in the ledger for tracking jobs applied externally. |
| `calibrate_selectors` | *none* | Discovers and calibrates DOM selectors against live pages with self-healing heuristics. |
| `search_linkedin_jobs` | `keywords: list[str]`, `location: str = "Israel"`, `limit: int = 25` | Dedicated LinkedIn search tool returning normalized Job models. |
| `get_linkedin_job_details` | `job_id: str` | Fetches rich job description, application URLs, and metadata for a specific LinkedIn posting. |
| `notify_new_jobs` | `jobs: list[dict]`, `channel: str = "telegram"` | Sends structured notification digest of new top-tier job opportunities. |
| `test_notifier` | `channel: str = "telegram"` | Tests notification channel configuration. |
| `set_operation_mode` | `mode: 'supervised' \| 'autonomous'` | Switches server execution mode between supervised and autonomous. |

---

## ⚡ Quick Start

### 1. Clone & Configure
```bash
git clone https://github.com/TechJobMCP/TechJobMCP.git
cd TechJobMCP

# Copy your CV and setup environment
cp /path/to/your/resume.pdf ./cv.pdf
cp .env.example .env
```

### 2. Run with Docker Compose
The Docker setup includes CPU-optimized PyTorch wheels and mounts local `./data` for the fine-tuned LAYA model and persistent ledger:

```bash
docker compose up -d
docker compose logs -f techjob-mcp
```
### 2.1. Export Public HTTPS Port for AI Clients
```bash
curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8000
```

Connect the generated `https://<tunnel-id>.trycloudflare.com/mcp` URL to your AI client. See [**AI Client Integrations Guide**](docs/CLIENT_INTEGRATIONS.md) for full setup instructions.

---



### 3. Dedicated CLI Tools
TechJobMCP provides purpose-built CLI scripts for autonomous scouting and evaluation:

```bash
# 1. Run complete end-to-end autonomous job hunt (Scout -> System 1 Score -> System 2 Tailor -> Auto-Apply)
uv run python scripts/run_autonomous_job_hunt.py --cv cv.pdf --limit 10

# 2. Scout real live job openings in Israel across all sources
uv run python scripts/scout_real_jobs.py --cv cv.pdf

# 3. Benchmark System 1 engines (Fine-Tuned LAYA vs. Base LAYA vs. Heuristic)
uv run python scripts/benchmark_system1_ab.py --sample-size 50
```

---

## 🧪 Running Tests

Run the full automated test suite (**948 unit and integration tests**):

```bash
uv run pytest
```

---

## 📄 License

This project is licensed under the MIT License.
