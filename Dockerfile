# ==========================================
# HireMeTech FastMCP Server Dockerfile
# Multi-stage Python 3.12 with Playwright
# ==========================================

FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency definition
COPY pyproject.toml README.md ./
COPY job_mcp ./job_mcp

# Build virtual environment
RUN uv venv /app/.venv && \
    uv pip install --no-cache -e .


FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime system dependencies for Chromium / Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv and application from builder
COPY --from=builder /app/.venv /app/.venv
COPY job_mcp ./job_mcp
COPY pyproject.toml README.md ./

# Put virtualenv on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BROWSER_HEADLESS=true \
    BROWSER_PROFILE_DIR=/app/browser_profile \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# Install Playwright browser binary
RUN playwright install chromium

# Create volume mount point for persistent browser profile
RUN mkdir -p /app/browser_profile
VOLUME ["/app/browser_profile"]

# Expose HTTP / SSE port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start MCP Server
CMD ["python", "-m", "job_mcp"]
