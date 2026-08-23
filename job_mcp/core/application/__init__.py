"""Application services for Tech Job MCP server."""

from job_mcp.core.application.dispatcher import (
    ISRAEL_LOCATION_PATTERN,
    HybridApplicationDispatcher,
)
from job_mcp.core.application.dom_inspector import (
    FormFieldSchema,
    SubmitButtonInfo,
    extract_form_schema,
    identify_submit_button,
)
from job_mcp.core.application.ledger_service import ApplicationLedger
from job_mcp.core.application.mapper import SemanticFormMapper
from job_mcp.core.application.strategies import (
    ApiPostStrategy,
    BrowserPlaywrightStrategy,
    EasyApplyStrategy,
)
from job_mcp.core.application.strategy import (
    ApplicationStrategy,
    get_application_strategy,
    register_application_strategy,
)

__all__ = [
    "ApplicationLedger",
    "HybridApplicationDispatcher",
    "ISRAEL_LOCATION_PATTERN",
    "SemanticFormMapper",
    "ApplicationStrategy",
    "get_application_strategy",
    "register_application_strategy",
    "ApiPostStrategy",
    "EasyApplyStrategy",
    "BrowserPlaywrightStrategy",
    "FormFieldSchema",
    "SubmitButtonInfo",
    "extract_form_schema",
    "identify_submit_button",
]

