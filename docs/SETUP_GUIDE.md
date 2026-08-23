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

# Resilient Free-Tier LLM Gateway (for screening questions)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-flash-lite-latest

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=google/gemini-2.0-flash-lite-preview-02-05:free

OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.2
LLM_CACHE_PATH=llm_cache.db

# Autonomous Application Safety Guardrails
AUTO_APPLY_ENABLED=false
MAX_DAILY_APPLICATIONS=10
APPLICATION_LEDGER_PATH=application_ledger.db

# Playwright Browser Automation
BROWSER_HEADLESS=true
BROWSER_PROFILE_DIR=./browser_profile

# Cache & Logging
CACHE_TTL_MINUTES=60
LOG_LEVEL=INFO
```

---

## 2. How Dynamic Skill & Target Role Extraction Works

When a CV is loaded, TechJobMCP dynamically extracts:
1. **Technical Skills (`profile.skills`)**: Discovers all relevant languages, frameworks, databases, and cloud tools from the CV text.
2. **Primary Stack (`profile.primary_stack`)**: Identifies the top recurring technologies in your experience (e.g., `Python, LangGraph, Azure, RAG` or `Java, Spring Boot, PostgreSQL, Docker`).
3. **Dynamic Target Roles (`profile.target_roles`)**: Automatically infers target job titles based on your specific combination of skills:

| Candidate CV Skills | Automatically Inferred Target Roles |
| :--- | :--- |
| **Java + Spring + Docker** | `Java Developer`, `Backend Engineer`, `Software Engineer` |
| **React + TypeScript + Next.js** | `Frontend Engineer`, `Full Stack Engineer`, `Software Engineer` |
| **Python + FastAPI + PostgreSQL** | `Python Developer`, `Backend Engineer`, `Software Engineer` |
| **Kubernetes + Terraform + AWS** | `DevOps Engineer`, `Cloud Engineer`, `Systems Engineer` |
| **PyTorch + LLM + LangGraph + RAG** | `AI Engineer`, `Machine Learning Engineer`, `Python Developer` |
| **Solidity + Foundry + Smart Contracts** | `Web3 Developer`, `Smart Contract Engineer`, `Blockchain Developer` |

### Overriding Target Roles & Preferences
You can always pass explicit target roles or keyword overrides when invoking MCP tools (like `run_job_scout` or `filter_jobs_by_preferences`):
```json
{
  "cv_path": "./cv.pdf",
  "location": "Israel",
  "keywords": ["AI Engineer", "Python Developer"],
  "exclude_keywords": ["Senior", "Lead", "5+ years", "US only"]
}
```

---

## 3. Running the Server

### Option A: Using Docker Compose (Recommended)

Docker runs the server on port 8000 inside an isolated container with Chromium, Ollama, and all dependencies pre-installed:

```bash
# Build and start container in detached mode
docker compose up -d

# View real-time logs
docker compose logs -f techjob-mcp

# Stop container
docker compose down
```

### Option B: Running Locally with Python (`uv` / virtualenv)

If you want to run directly in your local terminal:

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

## 4. Verifying Server Health

Once the server is running on port 8000, verify it locally:

```bash
# Check health endpoint
curl http://localhost:8000/health
# Output: {"status":"ok","server":"Tech Job MCP FastMCP Server"}

# Check MCP endpoint
curl -I http://localhost:8000/mcp
# Output: HTTP/1.1 200 OK
```

Next, see [**Tunneling & Port Export Guide**](./TUNNEL_AND_PORTS.md) to expose this port to external AI clients (Gemini Spark, Claude, ChatGPT).
