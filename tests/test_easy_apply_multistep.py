"""Unit tests for Multi-Step LinkedIn Easy Apply Strategy and Receipt Verification."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from job_mcp.core.application.strategies.easy_apply import EasyApplyStrategy
from job_mcp.models.ledger import ApplicationMethod
from job_mcp.models.schemas import CandidateProfile, Job, WorkMode


@pytest.fixture
def sample_job():
    return Job(
        job_id="test_linkedin_1",
        title="Junior Software Engineer",
        company="TechStart",
        location="Tel Aviv, Israel",
        work_mode=WorkMode.HYBRID,
        source="linkedin",
        url="https://www.linkedin.com/jobs/view/123456",
        apply_url="https://www.linkedin.com/jobs/view/123456",
    )


@pytest.fixture
def sample_profile():
    return CandidateProfile(
        full_name="Lior Zvieli",
        first_name="Lior",
        last_name="Zvieli",
        email="lior@example.com",
        phone="050-1234567",
        seniority_level="Junior",
        top_skills=["Python", "React", "Docker", "AI"],
    )


@pytest.mark.asyncio
async def test_easy_apply_preview(sample_job, sample_profile):
    """Verify EasyApplyStrategy preview returns proper metadata and fields."""
    strategy = EasyApplyStrategy()
    preview = await strategy.preview(sample_job, sample_profile, cv_path="cv.pdf")

    assert preview.job_id == sample_job.job_id
    assert preview.application_method == ApplicationMethod.EASY_APPLY.value
    assert preview.fields_to_submit.get("apply_mode") == "1_click_easy_apply"
    assert preview.fields_to_submit.get("cv_attached") is True


@pytest.mark.asyncio
async def test_easy_apply_multistep_submission(sample_job, sample_profile):
    """Verify multi-step modal traversal: Contact -> Experience -> Submit -> Receipt."""
    mock_session_manager = MagicMock()
    mock_page = AsyncMock()
    mock_session_manager.get_page = AsyncMock(return_value=mock_page)

    # Easy Apply button
    mock_easy_btn = AsyncMock()
    mock_easy_btn.count = AsyncMock(return_value=1)
    mock_easy_btn.is_visible = AsyncMock(return_value=True)
    mock_easy_btn.click = AsyncMock()
    mock_easy_btn.first = mock_easy_btn

    # Next button
    mock_next_btn = AsyncMock()
    mock_next_btn.count = AsyncMock(side_effect=[1, 1, 0, 0, 0])
    mock_next_btn.is_visible = AsyncMock(side_effect=[True, True, False, False, False])
    mock_next_btn.click = AsyncMock()
    mock_next_btn.first = mock_next_btn

    # Submit button
    mock_submit_btn = AsyncMock()
    mock_submit_btn.count = AsyncMock(side_effect=[0, 0, 1, 0, 0])
    mock_submit_btn.is_visible = AsyncMock(side_effect=[False, False, True, False, False])
    mock_submit_btn.click = AsyncMock()
    mock_submit_btn.first = mock_submit_btn

    # Confirmation element ("Application sent")
    mock_confirm_elem = AsyncMock()
    mock_confirm_elem.count = AsyncMock(side_effect=[0, 0, 0, 1, 1])
    mock_confirm_elem.is_visible = AsyncMock(side_effect=[False, False, False, True, True])
    mock_confirm_elem.first = mock_confirm_elem

    def mock_locator(selector):
        sel_lower = selector.lower()
        if "jobs-apply-button" in sel_lower or "easy apply" in sel_lower:
            return mock_easy_btn
        if "submit" in sel_lower or "שלח" in sel_lower:
            return mock_submit_btn
        if "next" in sel_lower or "continue" in sel_lower or "הבא" in sel_lower:
            return mock_next_btn
        if "application sent" in sel_lower:
            return mock_confirm_elem
        # Default empty locator
        empty = AsyncMock()
        empty.count = AsyncMock(return_value=0)
        empty.is_visible = AsyncMock(return_value=False)
        empty.first = empty
        return empty

    mock_page.locator = MagicMock(side_effect=mock_locator)
    mock_page.url = sample_job.apply_url

    strategy = EasyApplyStrategy(session_manager=mock_session_manager)

    with patch.object(strategy, "_capture_screenshot", new_callable=AsyncMock) as mock_shot:
        mock_shot.return_value = "data/screenshots/test_success.png"
        res = await strategy.apply(sample_job, sample_profile, cv_path="cv.pdf")

    assert res["success"] is True
    assert res["method"] == ApplicationMethod.EASY_APPLY.value
    assert res["status"] == "success"
    assert "receipt" in res
    assert res["receipt"]["confirmation_type"] == "LINKEDIN_APPLICATION_SENT"
    assert res["receipt"]["screenshot_path"] == "data/screenshots/test_success.png"
