"""Tests for structured observability logging."""

import json
import logging
import uuid

import pytest

from job_mcp.utils.logger import get_logger, generate_trace_id
from job_mcp.models.schemas import ToolResponse


class TestStructuredLogger:
    def test_get_logger_returns_bound_logger(self):
        log = get_logger("test.observability")
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
        assert hasattr(log, "bind")

    def test_logger_outputs_json(self, capfd):
        log = get_logger("test.json_output")
        log.info("test_event", key="value")
        captured = capfd.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l.strip()]
        assert lines, "No output captured on stderr"
        parsed = json.loads(lines[-1])
        assert parsed["event"] == "test_event"
        assert parsed["key"] == "value"
        assert "timestamp" in parsed

    def test_logger_sanitizes_secrets(self, capfd):
        log = get_logger("test.sanitize")
        log.info("auth_check", token="Bearer sk-abc123xyz789secret")
        captured = capfd.readouterr()
        line = captured.err.strip().split("\n")[-1]
        assert "sk-abc123xyz789secret" not in line
        assert "[REDACTED]" in line

    def test_logger_sanitizes_cookies(self, capfd):
        log = get_logger("test.sanitize_cookie")
        log.info("request", cookie="session=abc123;auth=xyz789")
        captured = capfd.readouterr()
        line = captured.err.strip().split("\n")[-1]
        assert "abc123" not in line
        assert "[REDACTED]" in line

    def test_logger_sanitizes_emails(self, capfd):
        log = get_logger("test.sanitize_email")
        log.info("user_login", email="user@example.com")
        captured = capfd.readouterr()
        line = captured.err.strip().split("\n")[-1]
        assert "user@example.com" not in line
        assert "[EMAIL_REDACTED]" in line

    def test_log_level_from_env(self, monkeypatch, capfd):
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        # Force re-creation of logger with new level
        log = get_logger("test.level_check_" + uuid.uuid4().hex[:6])
        log.info("should_not_appear")
        log.warning("should_appear", data="visible")
        captured = capfd.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l.strip()]
        # Only warning should appear
        assert any("should_appear" in l for l in lines)
        assert not any("should_not_appear" in l for l in lines)

    def test_logger_positional_args(self, capfd):
        log = get_logger("test.positional")
        log.info("Hello %s, number %d", "world", 123)
        captured = capfd.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l.strip()]
        parsed = json.loads(lines[-1])
        assert parsed["event"] == "Hello world, number 123"

    def test_logger_exception_stack_trace(self, capfd):
        log = get_logger("test.exception")
        try:
            raise ValueError("test exception for logger")
        except ValueError:
            log.exception("an error occurred")
        captured = capfd.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l.strip()]
        parsed = json.loads(lines[-1])
        assert parsed["event"] == "an error occurred"
        assert "exception" in parsed
        assert "ValueError: test exception for logger" in parsed["exception"]


class TestGenerateTraceId:
    def test_returns_hex_string(self):
        tid = generate_trace_id()
        assert isinstance(tid, str)
        assert len(tid) == 8
        int(tid, 16)  # Should not raise

    def test_uniqueness(self):
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100


class TestToolResponseTraceId:
    def test_trace_id_optional_default_none(self):
        resp = ToolResponse(success=True, message="ok")
        assert resp.trace_id is None

    def test_trace_id_set(self):
        resp = ToolResponse(success=True, message="ok", trace_id="abcd1234")
        assert resp.trace_id == "abcd1234"

    def test_trace_id_in_dump(self):
        resp = ToolResponse(success=True, message="ok", trace_id="ef567890")
        d = resp.model_dump()
        assert d["trace_id"] == "ef567890"
