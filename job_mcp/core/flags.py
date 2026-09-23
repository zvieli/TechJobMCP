"""Feature flags for TechJobMCP v2 dual-cognition architecture."""

import os

def is_flag_enabled(flag_name: str, default: bool = False) -> bool:
    """Check if a feature flag is enabled via environment variables."""
    val = os.getenv(flag_name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on", "enabled")

# Dual-cognition Laya System 1 feature flags (disabled by default in Phase 1)
TECHJOB_USE_LAYA_DEDUP: bool = is_flag_enabled("TECHJOB_USE_LAYA_DEDUP", default=False)
TECHJOB_USE_LAYA_SCORING: bool = is_flag_enabled("TECHJOB_USE_LAYA_SCORING", default=False)
TECHJOB_USE_LAYA_SECTIONS: bool = is_flag_enabled("TECHJOB_USE_LAYA_SECTIONS", default=False)
TECHJOB_USE_LAYA_FIELDS: bool = is_flag_enabled("TECHJOB_USE_LAYA_FIELDS", default=False)
