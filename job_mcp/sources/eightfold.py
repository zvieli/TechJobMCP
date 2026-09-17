"""Backward compatibility shim for job_mcp.sources.eightfold."""

import sys
from job_mcp.sources.public import eightfold as _impl

for _k, _v in _impl.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v

__all__ = getattr(_impl, "__all__", [k for k in globals() if not k.startswith("_")])

sys.modules[__name__] = _impl
