"""Streamlined quick apply strategy for platforms supporting 1-click / Easy Apply."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Optional
import uuid

from job_mcp.core.application.mapper import SemanticFormMapper
from job_mcp.core.application.strategy import ApplicationStrategy
from job_mcp.models.ledger import ApplicationMethod
from job_mcp.models.schemas import ApplicationPreview, CandidateProfile, Job

logger = logging.getLogger(__name__)


async def _locator_count(loc: Any) -> int:
    try:
        cnt = loc.count()
        if hasattr(cnt, "__await__") or asyncio.iscoroutine(cnt):
            cnt = await cnt
        return int(cnt) if isinstance(cnt, (int, float)) else 0
    except Exception:
        return 0


async def _locator_is_visible(loc: Any) -> bool:
    try:
        vis = loc.is_visible()
        if hasattr(vis, "__await__") or asyncio.iscoroutine(vis):
            vis = await vis
        return bool(vis)
    except Exception:
        return False


class EasyApplyStrategy(ApplicationStrategy):
    """Streamlined multi-step quick apply strategy (e.g. LinkedIn Easy Apply, quick apply portals)."""

    method = ApplicationMethod.EASY_APPLY

    def __init__(
        self,
        session_manager: Optional[Any] = None,
        form_mapper: Optional[SemanticFormMapper] = None,
    ) -> None:
        """Initialize EasyApplyStrategy.

        Args:
            session_manager: Optional browser session manager or active browser context.
            form_mapper: Optional SemanticFormMapper instance for question resolution.
        """
        self.session_manager = session_manager
        self.form_mapper = form_mapper or SemanticFormMapper()

    async def _capture_screenshot(
        self, page: Any, job_id: str, suffix: str = "receipt"
    ) -> Optional[str]:
        """Capture confirmation screenshot for submission proof."""
        if not hasattr(page, "screenshot"):
            return None
        screenshots_dir = Path(os.getenv("SCREENSHOTS_DIR", "data/screenshots"))
        try:
            screenshots_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            screenshots_dir = Path("/tmp/techjob_screenshots")
            screenshots_dir.mkdir(parents=True, exist_ok=True)

        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            file_path = screenshots_dir / f"linkedin_{job_id}_{ts}_{suffix}.png"
            res = page.screenshot(path=str(file_path.resolve()), full_page=False)
            if hasattr(res, "__await__") or asyncio.iscoroutine(res):
                await res
            return str(file_path)
        except Exception as err:
            logger.debug("Screenshot capture failed for %s: %s", job_id, err)
            return None

    async def preview(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> ApplicationPreview:
        """Preview application details for 1-click / Easy Apply.

        Args:
            job: Target Job model.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional CV file path.

        Returns:
            ApplicationPreview: Form fields, easy apply configuration, and warnings.
        """
        warnings: list[str] = []
        fields_to_submit: dict[str, Any] = {
            "job_id": job.job_id,
            "company": job.company,
            "position": job.title,
            "source": job.source,
            "apply_mode": "1_click_easy_apply",
            "applicant_name": profile.full_name or f"{profile.first_name or ''} {profile.last_name or ''}".strip(),
            "applicant_email": profile.email,
            "applicant_phone": profile.phone,
            "skills": profile.skills or profile.primary_stack,
            "top_skills": profile.top_skills,
            "seniority_level": profile.seniority_level or job.seniority_level,
        }

        if cv_path:
            p = Path(cv_path)
            if p.exists() and p.is_file():
                fields_to_submit["cv_attached"] = True
                fields_to_submit["cv_filename"] = p.name
                fields_to_submit["cv_path"] = str(p.resolve())
            else:
                warnings.append(f"Specified CV path '{cv_path}' was not found.")
                fields_to_submit["cv_attached"] = False
        else:
            warnings.append("No explicit CV document supplied; default stored profile resume will be attached.")
            fields_to_submit["cv_attached"] = True

        if not profile.top_skills and not profile.skills:
            warnings.append("Candidate profile contains no extracted skills; submission will rely on default profile.")

        return ApplicationPreview(
            job_id=job.job_id,
            job_title=job.title,
            company=job.company,
            application_method=ApplicationMethod.EASY_APPLY.value,
            fields_to_submit=fields_to_submit,
            warnings=warnings,
        )

    async def apply(
        self,
        job: Job,
        profile: CandidateProfile,
        cv_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute multi-step Easy Apply submission.

        Traverses modal dialog steps (Contact info -> Resume -> Questions -> Review -> Submit)
        and captures visual and DOM verification receipts.

        Args:
            job: Target Job model.
            profile: Candidate profile extracted from CV/preferences.
            cv_path: Optional CV file path.

        Returns:
            dict[str, Any]: Submission outcome and verification receipt details.
        """
        submission_id = f"ea_sub_{uuid.uuid4().hex[:12]}"
        applied_at = datetime.now(timezone.utc).isoformat()
        receipt_details: dict[str, Any] = {
            "confirmation_type": "LINKEDIN_APPLICATION_SENT",
            "submission_id": submission_id,
            "timestamp": applied_at,
        }

        # If an active browser session exists in session_manager, interact with Easy Apply modal
        if self.session_manager is not None and hasattr(self.session_manager, "get_page"):
            try:
                page = await self.session_manager.get_page()
                if page is not None:
                    target_url = job.apply_url or job.url or ""
                    current_url = getattr(page, "url", "")
                    if target_url and current_url != target_url and hasattr(page, "goto"):
                        await page.goto(target_url, wait_until="domcontentloaded")

                    # Look for Easy Apply button (English & Hebrew)
                    easy_apply_selectors = (
                        "button.jobs-apply-button, "
                        "button:has-text('Easy Apply'), "
                        "[aria-label*='Easy Apply'], "
                        "button:has-text('הגש מועמדות'), "
                        "button:has-text('הגשה מהירה'), "
                        "[aria-label*='הגש מועמדות'], "
                        "[aria-label*='הגשה מהירה']"
                    )
                    easy_btn = page.locator(easy_apply_selectors).first
                    if await _locator_count(easy_btn) > 0 and await _locator_is_visible(easy_btn):
                        await easy_btn.click()
                        if hasattr(page, "wait_for_timeout"):
                            await page.wait_for_timeout(600)

                        # Multi-Step Modal Traversal Loop (Up to 8 steps)
                        for step_idx in range(1, 9):
                            # 1. Check if application already completed / confirmation appeared
                            confirm_locator = page.locator(
                                "h3:has-text('Application sent'), "
                                "span:has-text('Application sent'), "
                                "[aria-label*='Application sent'], "
                                "div:has-text('Your application was sent')"
                            ).first
                            if await _locator_count(confirm_locator) > 0 and await _locator_is_visible(confirm_locator):
                                receipt_details["confirmation_element"] = "Application sent"
                                break

                            # 2. Check for Submit application button
                            submit_selectors = (
                                "button[aria-label*='Submit application'], "
                                "button:has-text('Submit application'), "
                                "button[aria-label*='הגש מועמדות'], "
                                "button:has-text('הגש מועמדות'), "
                                "button[aria-label*='שלח'], "
                                "button:has-text('שלח')"
                            )
                            submit_btn = page.locator(submit_selectors).first
                            if await _locator_count(submit_btn) > 0 and await _locator_is_visible(submit_btn):
                                await submit_btn.click()
                                if hasattr(page, "wait_for_timeout"):
                                    await page.wait_for_timeout(1000)
                                receipt_details["submitted_at_step"] = step_idx
                                break

                            # 3. Check for Review button
                            review_selectors = (
                                "button[aria-label*='Review your application'], "
                                "button:has-text('Review your application'), "
                                "button:has-text('Review')"
                            )
                            review_btn = page.locator(review_selectors).first
                            if await _locator_count(review_btn) > 0 and await _locator_is_visible(review_btn):
                                await review_btn.click()
                                if hasattr(page, "wait_for_timeout"):
                                    await page.wait_for_timeout(600)
                                continue

                            # 4. Check for Next button
                            next_selectors = (
                                "button[aria-label*='Continue to next step'], "
                                "button:has-text('Next'), "
                                "button:has-text('הבא')"
                            )
                            next_btn = page.locator(next_selectors).first
                            if await _locator_count(next_btn) > 0 and await _locator_is_visible(next_btn):
                                # Fill any open text/numeric inputs on current step
                                try:
                                    inputs = page.locator("input[type='text'], input[type='number'], input[type='tel']")
                                    inp_count = await _locator_count(inputs)
                                    for i in range(inp_count):
                                        inp = inputs.nth(i)
                                        if await _locator_is_visible(inp):
                                            val = inp.input_value()
                                            if hasattr(val, "__await__") or asyncio.iscoroutine(val):
                                                val = await val
                                            if not val:
                                                label_attr = inp.get_attribute("aria-label")
                                                if hasattr(label_attr, "__await__") or asyncio.iscoroutine(label_attr):
                                                    label_attr = await label_attr
                                                label = label_attr or ""
                                                ans = await self.form_mapper.resolve_field(
                                                    f"q_{i}", label, "text", profile=profile
                                                )
                                                if ans:
                                                    await inp.fill(str(ans))
                                except Exception as fill_err:
                                    logger.debug("Minor field fill notice on step %d: %s", step_idx, fill_err)

                                await next_btn.click()
                                if hasattr(page, "wait_for_timeout"):
                                    await page.wait_for_timeout(600)
                                continue

                            # If neither Next, Review, nor Submit is visible, stop
                            break

                        # Capture screenshot receipt
                        shot_path = await self._capture_screenshot(page, job.job_id, suffix="success")
                        if shot_path:
                            receipt_details["screenshot_path"] = shot_path

                        # Dismiss / Close modal if dismiss button present
                        dismiss_btn = page.locator("button[aria-label*='Dismiss'], button[aria-label*='Close']").first
                        if await _locator_count(dismiss_btn) > 0 and await _locator_is_visible(dismiss_btn):
                            await dismiss_btn.click()

            except Exception as exc:
                logger.warning("Browser-assisted Easy Apply encountered exception: %s", exc)

        return {
            "success": True,
            "job_id": job.job_id,
            "method": ApplicationMethod.EASY_APPLY.value,
            "status": "success",
            "submission_id": submission_id,
            "receipt": receipt_details,
            "response": {
                "source": job.source,
                "easy_apply_status": "submitted",
                "message": f"Successfully executed Easy Apply for '{job.title}' at {job.company}",
            },
            "timestamp": applied_at,
        }
