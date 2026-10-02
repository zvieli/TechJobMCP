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


# Hermetic company registry for the whole test session.
#
# `job_mcp.sources.registry` constructs provider instances at import time, and
# Milestone 5 providers resolve their company catalog from the active registry.
# Without this pin, a developer's own `portals.yml` in the checkout would be
# discovered while pytest *collects* unrelated test modules, so a typo in a
# personal config file could abort the entire run with CompanyRegistryError.
# Production code still fail-fasts on invalid config; only the test session is
# forced onto the built-in defaults. Individual tests exercise discovery and
# reload by calling resolve()/configure() explicitly.
#
# Discovery is therefore disabled for the session by setting
# COMPANY_REGISTRY_BUILTINS_ONLY *before* anything under `job_mcp` is imported:
# importing job_mcp builds the provider registry, and that resolution reads the
# project config. Setting the flag afterwards is too late.
def _pin_builtin_company_registry() -> None:
    os.environ["COMPANY_REGISTRY_BUILTINS_ONLY"] = "1"
    os.environ.pop("COMPANY_REGISTRY_PATH", None)


_pin_builtin_company_registry()


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
