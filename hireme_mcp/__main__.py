"""Entry point for python -m hireme_mcp."""

from __future__ import annotations

import os
import sys
import uvicorn

from hireme_mcp.main import GeminiProbeMiddleware, mcp
from hireme_mcp.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the HireMeTech FastMCP server with configured transport."""
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    host = os.getenv("MCP_HOST", "0.0.0.0").strip()
    port_str = os.getenv("MCP_PORT", "8000").strip()

    try:
        port = int(port_str)
    except ValueError:
        logger.warning("Invalid MCP_PORT '%s', defaulting to 8000.", port_str)
        port = 8000

    logger.info(
        "Starting HireMeTech FastMCP Server (transport=%s, host=%s, port=%d)",
        transport,
        host,
        port,
    )

    if transport in ("http", "sse", "streamable-http"):
        http_transport = "sse" if transport == "sse" else "http"
        app = mcp.http_app(transport=http_transport)
        app.add_middleware(GeminiProbeMiddleware)
        uvicorn.run(app, host=host, port=port)
    elif transport == "stdio":
        mcp.run(transport="stdio")
    else:
        logger.error("Unsupported transport protocol: '%s'. Supported: stdio, http, sse", transport)
        sys.exit(1)


if __name__ == "__main__":
    main()
