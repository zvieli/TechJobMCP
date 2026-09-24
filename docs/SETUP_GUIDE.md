# TechJobMCP Setup & Configuration Guide

This guide explains how to configure and run the **TechJobMCP** server locally for any candidate profile, customize dynamic skill and target role extraction, and run the server using Docker Compose or native Python.

---

## 1. Candidate Profile & CV Configuration

TechJobMCP is built to serve **any candidate profile** across all technical disciplines (e.g., Python, Java, Full Stack, Frontend, DevOps, Cloud, AI/ML, Web3, Data).

### Supported Resume Formats
Place your CV/resume in the project root directory or any accessible path:
- **PDF**: `cv.pdf`, `resume.pdf`, `my_cv.pdf` (parsed via `pypdf` / `pymupdf`)
- **Word Document**: `cv.docx`, `resume.docx` (parsed via built-in XML extractor)
- **Plain Text**: `cv.txt`, `resume.txt`

### Setting up Environment Variables (`.env`)

Copy the example environment configuration:
```bash
cp .env.example .env
```

Edit `.env` with your details:
```ini
# Candidate & CV Configuration
DEFAULT_CV_PATH=./cv.pdf
CANDIDATE_NAME="Your Full Name"
CANDIDATE_EMAIL=your.email@example.com

# FastMCP Transport Configuration
# Use 'http' for web/remote/cloud clients, 'stdio' for desktop-only clients
MCP_TRANSPORT=http
MCP_HOST=0.0.0.0
MCP_PORT=8000

# System 1: Neural Triage Engine (LAYA)
# Options: laya (fine-tuned Llama-3.2-3B), base_laya (zero-shot), generative (LLM API), heuristic (deterministic)
SYSTEM1_ENGINE=laya
LAYA_MODEL_PATH=./data/models/laya-techjob

# System 2: Cognitive Application Tailoring Engine
ENABLE_SYSTEM2_AUTO_TAILOR=true

# Resilient Free-Tier LLM Gateway (for screening questions & System 2 tailoring)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-flash-lite-latest

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=google/gemini-2.0-flash-lite-preview-02-05:free

OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.2
LLM_CACHE_PATH=./data/llm_cache.db

# Autonomous Application Safety Guardrails
AUTO_APPLY_ENABLED=false
MAX_DAILY_APPLICATIONS=10
APPLICATION_LEDGER_PATH=./data/application_ledger.db

# Pluggable Source Toggles (All 11 Sources)
ENABLE_COMEET=true
ENABLE_GREENHOUSE=true
ENABLE_LEVER=true
ENABLE_WORKDAY=true
ENABLE_EIGHTFOLD=true
ENABLE_DIRECT_TECH=true
ENABLE_LINKEDIN=true
ENABLE_HIREMETECH=true
ENABLE_JOBIFY=true
ENABLE_GOTFRIENDS=true
ENABLE_ALLJOBS=false

# Playwright Browser Automation & Receipts
BROWSER_HEADLESS=true
BROWSER_PROFILE_DIR=./browser_profile
RECEIPTS_DIR=./data/receipts

# Cache & Logging
CACHE_TTL_MINUTES=60
LOG_LEVEL=INFO
```

---

## 2. System 1: Neural Triage Engine (LAYA)

TechJobMCP incorporates a dual-system cognitive architecture:
- **System 1 (Fast, Neural)**: Evaluates candidate-job semantic alignment in **<1ms**, instantly filtering out hundreds of non-matching postings and enforcing strict seniority constraints.
- **System 2 (Deliberative, LLM)**: Deeply analyzes high-confidence matches ($\ge 70-85$) to draft custom cover letters, CV highlights, recruiter pitches, and interview prep questions.

### Engine Options (`SYSTEM1_ENGINE`)
| Engine | Backend | Latency | Recommended Use |
|---|---|---|---|
| `laya` *(default)* | Fine-Tuned LAYA (Llama-3.2-3B) | **<1 ms** | Production autonomous triage with fine-tuned domain weights |
| `base_laya` | Base Llama-3.2-3B | ~25 ms | Baseline comparison and zero-shot A/B benchmarking |
| `generative` | Gemini Flash Lite / OpenRouter | 500–1200 ms | Cloud LLM evaluation when local GPU/CPU is unavailable |
| `heuristic` | Deterministic Token Scorer | <0.1 ms | Fallback rule-based matching with keyword weights |

### Setting Up LAYA Model Weights
Place the fine-tuned model checkpoint inside `./data/models/laya-techjob`:
```bash
mkdir -p data/models/laya-techjob
# The model directory should contain config.json, model.safetensors, tokenizer.json, etc.
```

When running with Docker Compose, `./data` is automatically mounted into `/app/data`, allowing the container to load weights seamlessly on both CPU and CUDA environments.

---

## 3. System 2: Application Tailoring Engine

When `ENABLE_SYSTEM2_AUTO_TAILOR=true`, the server automatically tailors every qualified application before submission:
1. **Targeted CV Highlights**: Extracts and rephrases the 3–5 most relevant achievements from your resume matching the target job description.
2. **Bilingual Cover Letters**: Generates a professional cover letter in English or Hebrew, matching the job posting's language and tone.
3. **Recruiter Outreach Pitch**: Produces a concise 2–3 sentence LinkedIn/email pitch tailored to the hiring team.
4. **Anticipated Interview Questions**: Analyzes the delta between your profile and job requirements to anticipate technical and behavioral interview questions.

---

## 4. Universal Auto-Apply & 4-Layer Verification Receipts

TechJobMCP supports 11 job sources and provides verifiable, cryptographic-level proof for every submitted application.

### Supported Strategies
- **LinkedIn Easy Apply (`EasyApplyStrategy`)**: Traverses multi-step applications (up to 8 steps), mapping contact details, work experience, file attachments, and answering screening questions using the resilient LLM gateway.
- **Universal Headless Browser (`BrowserPlaywrightStrategy`)**: Traverses ATS forms across Comeet (including `iframe#comeet-iframe` embedding), Greenhouse, Lever, Workday CXS, Eightfold, DirectTech, and regional portals.
- **Direct API Post (`ApiPostStrategy`)**: Fast JSON submission where REST endpoints are available.

### 4-Layer Verification Receipts
Every submission captures:
- **Layer 1 (DOM Proof)**: Detects confirmation text (e.g., "Thank you for applying", "Application submitted").
- **Layer 2 (Redirect URL)**: Validates navigation to confirmation URLs.
- **Layer 3 (Network Interception)**: Captures HTTP 200/201 responses from ATS submission endpoints.
- **Layer 4 (Screenshot Proof)**: Saves a timestamped screenshot to `./data/receipts/proof_<job_id>_<timestamp>.png`.

All receipts are recorded in the SQLite audit ledger (`./data/application_ledger.db`).

---

## 5. Dynamic Candidate Skill & Target Role Extraction

When a CV is loaded, TechJobMCP dynamically extracts:
1. **Technical Skills (`profile.skills`)**: Discovers all relevant languages, frameworks, databases, and cloud tools from the CV text.
2. **Primary Stack (`profile.primary_stack`)**: Identifies the top recurring technologies in your experience (e.g., `Python, LangGraph, Azure, RAG` or `Java, Spring Boot, PostgreSQL, Docker`).
3. **Seniority Level**: Inferred level (Student, Junior, Mid, Senior, Lead) used by System 1 to cap mismatched positions.
4. **Dynamic Target Roles (`profile.target_roles`)**: Automatically infers target job titles based on your specific combination of skills:

| Candidate CV Skills | Automatically Inferred Target Roles |
| :--- | :--- |
| **Java + Spring + Docker** | `Java Developer`, `Backend Engineer`, `Software Engineer` |
| **React + TypeScript + Next.js** | `Frontend Engineer`, `Full Stack Engineer`, `Software Engineer` |
| **Python + FastAPI + PostgreSQL** | `Python Developer`, `Backend Engineer`, `Software Engineer` |
| **Kubernetes + Terraform + AWS** | `DevOps Engineer`, `Cloud Engineer`, `Systems Engineer` |
| **PyTorch + LLM + LangGraph + RAG** | `AI Engineer`, `Machine Learning Engineer`, `Python Developer` |
| **Solidity + Foundry + Smart Contracts** | `Web3 Developer`, `Smart Contract Engineer`, `Blockchain Developer` |

---

## 6. Running the Server

### Option A: Using Docker Compose (Recommended)

Docker runs the server on port 8000 inside an isolated container with Chromium, CPU-optimized PyTorch, and persistent volume mounting:

```bash
# Build and start container in detached mode
docker compose up -d

# View real-time logs
docker compose logs -f techjob-mcp

# Stop container
docker compose down
```

### Option B: Running Locally with Python (`uv`)

```bash
# 1. Create virtual environment and install dependencies
uv venv .venv
uv pip install -e ".[dev]"
playwright install chromium

# 2. Start FastMCP Server on HTTP transport (Port 8000)
uv run python -m job_mcp --transport http --host 0.0.0.0 --port 8000

# 3. Or start in Stdio transport for local desktop clients:
uv run python -m job_mcp --transport stdio
```

---

## 7. Dedicated CLI Scripts

TechJobMCP includes CLI utilities for autonomous workflows and evaluation:

```bash
# 1. Complete Autonomous Job Hunt (Scout -> System 1 Score -> System 2 Tailor -> Auto-Apply)
uv run python scripts/run_autonomous_job_hunt.py --cv cv.pdf --limit 10

# 2. Live Scouting for positions in Israel across all 11 sources
uv run python scripts/scout_real_jobs.py --cv cv.pdf

# 3. System 1 A/B Evaluation Benchmark (Fine-Tuned LAYA vs Base LAYA vs Heuristic)
uv run python scripts/benchmark_system1_ab.py --sample-size 50
```

---

## 8. Verifying Server Health

```bash
# Check health endpoint
curl http://localhost:8000/health
# Output: {"status":"ok","server":"Tech Job MCP FastMCP Server"}

# Check MCP endpoint
curl -I http://localhost:8000/mcp
# Output: HTTP/1.1 200 OK
```

Next, see [**Tunneling & Port Export Guide**](./TUNNEL_AND_PORTS.md) to expose this port to external AI clients (Gemini Spark, Claude, ChatGPT).
