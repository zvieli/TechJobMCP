"""Unit and integration tests for platform-specific ATS adapters and multi-layer receipts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from job_mcp.core.application.dom_inspector import FormFieldSchema, SubmitButtonInfo
from job_mcp.core.application.strategies.browser import BrowserPlaywrightStrategy
from job_mcp.models.ledger import ApplicationMethod
from job_mcp.models.schemas import CandidateProfile, Job


@pytest.fixture
def candidate_with_cover_letter():
    return CandidateProfile(
        full_name="Lior Zvieli",
        first_name="Lior",
        last_name="Zvieli",
        email="lior@example.com",
        phone="050-1234567",
        skills=["Python", "FastAPI", "React", "AI"],
        top_skills=["Python", "FastAPI"],
        cover_letter="Dear Hiring Manager,\nI am writing to express my strong enthusiasm for this engineering role at your company.\nBest,\nLior",
    )


@pytest.mark.asyncio
async def test_comeet_iframe_detection_and_receipt(candidate_with_cover_letter):
    """Test Comeet iframe detection and confirmation receipt handling."""
    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://careers.example.com/jobs/comeet-role"
    mock_page.title = AsyncMock(return_value="Careers - Comeet Role")
    mock_page.screenshot = AsyncMock()
    mock_page.content = AsyncMock(return_value="<div>Thank you for applying to Comeet!</div>")
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    # Mock Comeet iframe locator
    mock_iframe_loc = MagicMock()
    mock_iframe_loc.count = AsyncMock(return_value=1)
    mock_iframe_loc.first = mock_iframe_loc

    # Mock submit button locator
    mock_submit_loc = MagicMock()
    mock_submit_loc.count = AsyncMock(return_value=1)
    mock_submit_loc.first = mock_submit_loc
    mock_submit_loc.click = AsyncMock()

    # Mock form input locators
    mock_input_loc = MagicMock()
    mock_input_loc.count = AsyncMock(return_value=1)
    mock_input_loc.fill = AsyncMock()

    def mock_locator(sel):
        sel_lower = sel.lower()
        if "comeet" in sel_lower:
            return mock_iframe_loc
        if "submit" in sel_lower or "apply" in sel_lower:
            return mock_submit_loc
        return mock_input_loc

    mock_page.locator = MagicMock(side_effect=mock_locator)

    comeet_job = Job(
        job_id="comeet_comm_it_101",
        title="Software Engineer",
        company="Comm-IT",
        source="comeet",
        apply_url="https://careers.example.com/jobs/comeet-role",
    )

    strategy = BrowserPlaywrightStrategy(session_manager=mock_session_manager)

    mock_fields = [
        FormFieldSchema(
            field_id="c_name",
            name="name",
            field_type="text",
            label="Full Name",
            selector="input[name='name']",
        ),
        FormFieldSchema(
            field_id="c_email",
            name="email",
            field_type="email",
            label="Email",
            selector="input[name='email']",
        ),
        FormFieldSchema(
            field_id="c_note",
            name="note",
            field_type="textarea",
            label="Personal Note / Cover Letter",
            selector="textarea[name='note']",
        ),
    ]

    mock_submit_info = SubmitButtonInfo(
        selector="button[type='submit']",
        text="Submit Application",
        confidence=0.95,
    )

    with (
        patch("job_mcp.core.application.strategies.browser.extract_form_schema", AsyncMock(return_value=mock_fields)),
        patch("job_mcp.core.application.strategies.browser.identify_submit_button", AsyncMock(return_value=mock_submit_info)),
    ):
        result = await strategy.apply(comeet_job, candidate_with_cover_letter)

    assert result["success"] is True
    assert result["status"] == "success"
    assert result["method"] == ApplicationMethod.BROWSER.value
    assert "receipt_details" in result
    assert result["receipt_details"]["confirmed"] is True
    assert result["receipt_details"]["confirmation_type"] == "DOM_CONFIRMATION"
    assert "thank you for applying" in result["receipt_details"]["receipt_text"].lower()


@pytest.mark.asyncio
async def test_network_http_receipt_layer(candidate_with_cover_letter):
    """Test network listener capturing HTTP 200 on submission endpoint as Layer 3 proof."""
    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://boards.greenhouse.io/techcorp/jobs/123"
    mock_page.title = AsyncMock(return_value="Greenhouse Application")
    mock_page.screenshot = AsyncMock()
    # Content has no obvious phrase, URL didn't change
    mock_page.content = AsyncMock(return_value="<div>Processing...</div>")
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    mock_submit_loc = MagicMock()
    mock_submit_loc.count = AsyncMock(return_value=1)
    mock_submit_loc.first = mock_submit_loc
    mock_submit_loc.click = AsyncMock()

    mock_input_loc = MagicMock()
    mock_input_loc.count = AsyncMock(return_value=1)
    mock_input_loc.fill = AsyncMock()

    mock_page.locator = MagicMock(return_value=mock_input_loc)

    event_handlers = {}
    def mock_on(event, handler):
        event_handlers[event] = handler

    mock_page.on = MagicMock(side_effect=mock_on)

    greenhouse_job = Job(
        job_id="gh_123",
        title="Senior Python Engineer",
        company="TechCorp",
        source="greenhouse",
        apply_url="https://boards.greenhouse.io/techcorp/jobs/123",
    )

    strategy = BrowserPlaywrightStrategy(session_manager=mock_session_manager)

    mock_submit_info = SubmitButtonInfo(
        selector="button#submit_app",
        text="Submit Application",
        confidence=0.95,
    )

    # When click is called on submit, simulate an HTTP response emitted to the page
    async def simulate_submit_click():
        if "response" in event_handlers:
            resp_mock = MagicMock()
            resp_mock.status = 200
            resp_mock.url = "https://boards-api.greenhouse.io/v1/applications"
            event_handlers["response"](resp_mock)

    mock_submit_loc.click = AsyncMock(side_effect=simulate_submit_click)

    def locator_router(sel):
        if "submit" in sel.lower():
            return mock_submit_loc
        return mock_input_loc

    mock_page.locator = MagicMock(side_effect=locator_router)

    mock_fields = [
        FormFieldSchema(
            field_id="first_name",
            name="first_name",
            field_type="text",
            selector="input#first_name",
        )
    ]

    with (
        patch("job_mcp.core.application.strategies.browser.extract_form_schema", AsyncMock(return_value=mock_fields)),
        patch("job_mcp.core.application.strategies.browser.identify_submit_button", AsyncMock(return_value=mock_submit_info)),
    ):
        result = await strategy.apply(greenhouse_job, candidate_with_cover_letter)

    assert result["success"] is True
    assert result["status"] == "success"
    assert "receipt_details" in result
    assert result["receipt_details"]["confirmed"] is True
    assert result["receipt_details"]["confirmation_type"] == "HTTP_STATUS_200"
    assert "HTTP 200" in result["receipt_details"]["receipt_text"]
    assert "greenhouse.io" in result["receipt_details"]["receipt_text"]
