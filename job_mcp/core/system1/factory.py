from typing import Optional
from job_mcp.core.system1.interface import System1Engine
from job_mcp.core.system1.engine import LazyLayaEngine

_ACTIVE_ENGINE: Optional[System1Engine] = None

def set_active_engine(engine: System1Engine) -> None:
    """Inject a custom System1Engine implementation (e.g. GenerativeBaselineEngine for A/B testing)."""
    global _ACTIVE_ENGINE
    _ACTIVE_ENGINE = engine

def get_system1_engine() -> System1Engine:
    """Get the active System 1 engine, defaulting to the fine-tuned LAYA model."""
    global _ACTIVE_ENGINE
    if _ACTIVE_ENGINE is None:
        _ACTIVE_ENGINE = LazyLayaEngine.get_instance()
    return _ACTIVE_ENGINE


def reset_engine() -> None:
    """Reset the active engine to default (None)."""
    global _ACTIVE_ENGINE
    _ACTIVE_ENGINE = None

