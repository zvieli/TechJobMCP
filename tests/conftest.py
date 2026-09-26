import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _ROOT / ".scripts"
if _SCRIPTS_DIR.is_dir() and "scripts" not in sys.modules:
    _pkg = types.ModuleType("scripts")
    _pkg.__path__ = [str(_SCRIPTS_DIR)]
    sys.modules["scripts"] = _pkg

"""Pytest configuration and environment isolation fixtures."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch):
    """Sanitize ambient container/daemon environment variables during test runs.

    Prevents tests without explicit CV or API credentials from picking up
    live container paths (/app/cv.pdf) or real OpenRouter/Gemini API keys.
    """
    vars_to_clear = [
        "DEFAULT_CV_PATH",
        "BROWSER_PROFILE_DIR",
        "CACHE_TTL_MINUTES",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ]
    for var in vars_to_clear:
        if var in os.environ:
            monkeypatch.delenv(var, raising=False)
