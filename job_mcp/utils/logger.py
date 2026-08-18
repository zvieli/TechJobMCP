"""Structured JSON logging with secret sanitization for HireMeTech MCP server."""

import logging
import os
import re
import sys
import uuid
from typing import Any

import structlog


# ──────────────────────────────────────────────
# Trace ID generation
# ──────────────────────────────────────────────

def generate_trace_id() -> str:
    """Generate an 8-character hex trace ID."""
    return uuid.uuid4().hex[:8]


# ──────────────────────────────────────────────
# Secret sanitization processor (structlog)
# ──────────────────────────────────────────────

_SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_token",
    "auth",
    "auth_token",
    "client_secret",
    "cookie",
    "cookies",
}

_REDACTION_RULES: list[tuple[re.Pattern, str]] = [
    # Authorization header / Bearer tokens
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE), r"\1[REDACTED]"),
    # Key-value secrets (password, token, secret, api_key, auth, cookie, session)
    (
        re.compile(
            r'(?i)(\"?\b(?:password|passwd|secret|token|api_?key|access_?token|auth(?:_?token)?|client_?secret|session|cookie)\"?\s*[:=]\s*[\"\'?]?)((?:(?![\"\'\s,;&]).)+)([\"\'?]?)',
            re.IGNORECASE,
        ),
        r"\1[REDACTED]\3",
    ),
    # Cookie / Set-Cookie header patterns
    (re.compile(r"(?i)(cookie\s*[:=]\s*)[^\r\n]+", re.IGNORECASE), r"\1[REDACTED]"),
    # Email addresses
    (
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            re.IGNORECASE,
        ),
        "[EMAIL_REDACTED]",
    ),
]


def _sanitize_value(value: Any) -> Any:
    """Apply redaction rules to a value recursively."""
    if isinstance(value, str):
        sanitized = value
        for pattern, replacement in _REDACTION_RULES:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized
    elif isinstance(value, dict):
        return {
            k: ("[REDACTED]" if str(k).lower() in _SENSITIVE_KEYS else _sanitize_value(v))
            for k, v in value.items()
        }
    elif isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return value


def sanitize_processor(
    logger: Any, method_name: str, event_dict: dict
) -> dict:
    """Structlog processor that sanitizes sensitive data and keys in the event dict."""
    for key in list(event_dict.keys()):
        if str(key).lower() in _SENSITIVE_KEYS:
            event_dict[key] = "[REDACTED]"
        else:
            event_dict[key] = _sanitize_value(event_dict[key])
    return event_dict


_LEVEL_ORDER = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "exception": logging.ERROR,
}


def _level_filter_processor(
    logger: Any, method_name: str, event_dict: dict
) -> dict:
    """Filter log events based on current LOG_LEVEL environment variable."""
    current_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    current_level = getattr(logging, current_level_name, logging.INFO)
    method_level = _LEVEL_ORDER.get(str(method_name).lower(), logging.INFO)
    if method_level < current_level:
        raise structlog.DropEvent
    return event_dict


class _StderrLogger(structlog.PrintLogger):
    """Print logger that always prints to the current sys.stderr."""

    def __init__(self) -> None:
        super().__init__(file=sys.stderr)

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)


class _StderrLoggerFactory:
    """Factory producing _StderrLogger instances."""

    def __call__(self, *args: Any, **kwargs: Any) -> _StderrLogger:
        return _StderrLogger()


# ──────────────────────────────────────────────
# One-time structlog configuration
# ──────────────────────────────────────────────

_configured = False


def _configure_structlog() -> None:
    """Configure structlog once globally."""
    global _configured
    if _configured:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(remove_positional_args=True),
            _level_filter_processor,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            sanitize_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=_StderrLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    _configured = True


# ──────────────────────────────────────────────
# Public API — drop-in replacement
# ──────────────────────────────────────────────

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured JSON logger with secret sanitization.

    Drop-in replacement for the previous get_logger — callers use the same
    .info(), .warning(), .error(), .exception() interface.
    """
    _configure_structlog()
    return structlog.get_logger(name)
