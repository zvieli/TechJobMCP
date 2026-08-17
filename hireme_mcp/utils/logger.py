"""Sanitizing logging utility for HireMeTech MCP server."""

import logging
import os
import re


class SanitizingFormatter(logging.Formatter):
    """Log formatter that redacts sensitive information such as tokens, passwords, cookies, and emails."""

    REDACTION_PATTERNS = [
        # Authorization header / Bearer tokens
        (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE), r"\1[REDACTED]"),
        # Key-value secrets (password, token, secret, api_key, auth, cookie)
        (
            re.compile(
                r'(?i)("?(?:password|passwd|secret|token|api_?key|access_?token|auth(?:_?token)?|client_?secret)"?\s*[:=]\s*["\']?)(?:[^"\'\s,;&]+)(["\']?)',
                re.IGNORECASE,
            ),
            r"\1[REDACTED]\2",
        ),
        # Cookie / Set-Cookie header patterns
        (
            re.compile(r"(?i)(cookie\s*[:=]\s*)[^\r\n]+", re.IGNORECASE),
            r"\1[REDACTED]",
        ),
        # Email addresses
        (
            re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                re.IGNORECASE,
            ),
            "[EMAIL_REDACTED]",
        ),
    ]

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        sanitized = original
        for pattern, replacement in self.REDACTION_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized


def get_logger(name: str) -> logging.Logger:
    """Get or create a configured logger with sensitive data sanitization."""
    logger = logging.getLogger(name)
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level_name, logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = SanitizingFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
