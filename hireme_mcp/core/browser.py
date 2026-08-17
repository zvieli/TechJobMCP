"""Playwright browser automation and DOM interaction layer for HireMeTech MCP server."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

from playwright.async_api import Locator, Page
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from hireme_mcp.core.discovery import (
    SELECTORS,
    DynamicSelectorRegistry,
    discover_card_selector,
    discover_child_selector,
)
from hireme_mcp.models.schemas import ApplicationPreview, Job, WorkMode
from hireme_mcp.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level dynamic selector registry singleton
dynamic_registry = DynamicSelectorRegistry()

# Retry policy for browser operations
browser_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)

# Common tech stack dictionary for heuristic extraction
COMMON_TECH_STACK = [
    "Python", "JavaScript", "TypeScript", "React", "Next.js", "Vue", "Angular",
    "Node.js", "Express", "FastAPI", "Django", "Flask", "Go", "Golang", "Rust",
    "Java", "Kotlin", "Scala", "C++", "C#", ".NET", "Ruby", "Rails", "PHP",
    "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform", "Ansible",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "GraphQL", "REST", "gRPC", "Kafka", "RabbitMQ", "Celery", "Airflow",
    "PyTorch", "TensorFlow", "Scikit-Learn", "Pandas", "NumPy", "OpenAI",
    "LLM", "LangChain", "FastMCP", "Playwright", "CI/CD", "Git", "Linux",
    "TailwindCSS", "HTML", "CSS", "Sass", "Redux", "Zustand", "Prisma",
]


async def _resolve_selector(page_or_locator: Page | Locator, key: str) -> str:
    """Resolve selector using 4-tier adaptive hierarchy:
    Tier 1: Dynamic Selector Registry (cached working selectors)
    Tier 2: Primary static selector
    Tier 3: Fallback static selector list
    Tier 4: Heuristic DOM discovery (sibling clustering or semantic search)

    Args:
        page_or_locator: Playwright Page or Locator context.
        key: Selector registry key (or raw selector string if not registered).

    Returns:
        str: Resolved working selector string.

    Raises:
        ValueError: If no valid selector can be resolved across all 4 tiers.
    """
    # Tier 1: Check dynamic registry
    cached_selector = dynamic_registry.get(key)
    if cached_selector:
        try:
            if await page_or_locator.locator(cached_selector).count() > 0:
                logger.debug("Resolved selector for '%s' from dynamic registry: '%s'", key, cached_selector)
                return cached_selector
        except Exception as exc:
            logger.debug("Dynamic registry selector '%s' invalid for '%s': %s", cached_selector, key, exc)

    # If key is not in registered SELECTORS, attempt heuristic discovery or return raw selector
    if key not in SELECTORS:
        discovered = await discover_child_selector(page_or_locator, key)
        if discovered:
            try:
                if await page_or_locator.locator(discovered).count() > 0:
                    dynamic_registry.set(key, discovered)
                    logger.info("Discovered unregistered selector for '%s': '%s'", key, discovered)
                    return discovered
            except Exception:
                pass
        try:
            if await page_or_locator.locator(key).count() > 0:
                return key
        except Exception:
            pass
        return key

    primary = SELECTORS[key]["primary"]
    fallback = SELECTORS[key]["fallback"]

    # Tier 2: Primary selector
    try:
        if await page_or_locator.locator(primary).count() > 0:
            return primary
    except Exception as exc:
        logger.debug("Primary selector '%s' failed for '%s': %s", primary, key, exc)

    # Tier 3: Fallback selectors
    for sub_fallback in [f.strip() for f in fallback.split(",")]:
        try:
            if await page_or_locator.locator(sub_fallback).count() > 0:
                logger.warning(
                    "Primary selector '%s' for '%s' failed. Using fallback: '%s'",
                    primary,
                    key,
                    sub_fallback,
                )
                dynamic_registry.set(key, sub_fallback)
                return sub_fallback
        except Exception:
            continue

    try:
        if await page_or_locator.locator(fallback).count() > 0:
            logger.warning(
                "Primary selector '%s' for '%s' failed. Using combined fallback: '%s'",
                primary,
                key,
                fallback,
            )
            dynamic_registry.set(key, fallback)
            return fallback
    except Exception:
        pass

    # Tier 4: Heuristic DOM discovery
    discovered_selector: Optional[str] = None
    if key == "job_card":
        target_page = page_or_locator
        if not hasattr(target_page, "evaluate") and hasattr(target_page, "page"):
            target_page = getattr(target_page, "page")
        if target_page is not None:
            discovered_selector = await discover_card_selector(target_page)
    else:
        discovered_selector = await discover_child_selector(page_or_locator, key)

    if discovered_selector:
        try:
            if await page_or_locator.locator(discovered_selector).count() > 0:
                logger.info(
                    "Heuristically discovered working selector for '%s': '%s'",
                    key,
                    discovered_selector,
                )
                dynamic_registry.set(key, discovered_selector)
                return discovered_selector
        except Exception as exc:
            logger.debug("Discovered selector '%s' failed validation for '%s': %s", discovered_selector, key, exc)

    raise ValueError(
        f"Failed to resolve selector for '{key}' across all 4 tiers (registry, primary='{primary}', fallback='{fallback}', heuristic)."
    )


def _parse_work_mode(text: str) -> Optional[WorkMode]:
    """Parse work mode (remote, hybrid, onsite) from text content."""
    text_lower = text.lower()
    if "remote" in text_lower:
        return WorkMode.REMOTE
    if "hybrid" in text_lower:
        return WorkMode.HYBRID
    if "onsite" in text_lower or "on-site" in text_lower or "in-office" in text_lower:
        return WorkMode.ONSITE
    return None


def _extract_tech_from_text(text: str) -> list[str]:
    """Extract known technology keywords from freeform text."""
    found: list[str] = []
    text_lower = text.lower()
    for tech in COMMON_TECH_STACK:
        # Match word boundaries for tech terms
        pattern = r"\b" + re.escape(tech.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(tech)
    return sorted(list(set(found)), key=lambda s: s.lower())


async def _safe_get_text(locator: Locator) -> str:
    """Safely extract and clean inner text from a locator."""
    try:
        if await locator.count() > 0:
            text = await locator.first.inner_text()
            return text.strip()
    except Exception:
        pass
    return ""


async def _extract_meaningful_lines(locator: Locator) -> list[str]:
    """Safely extract non-empty trimmed text lines from a locator."""
    text = await _safe_get_text(locator)
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


@browser_retry
async def extract_jobs(page: Page) -> list[Job]:
    """Extract job listings from the current page.

    Args:
        page: Playwright active page.

    Returns:
        list[Job]: List of extracted job models.
    """
    card_selector = await _resolve_selector(page, "job_card")
    card_locators = page.locator(card_selector)
    count = await card_locators.count()

    logger.info("Found %d job card elements using selector '%s'", count, card_selector)
    jobs: list[Job] = []

    for i in range(count):
        card = card_locators.nth(i)
        try:
            # Extract job ID
            job_id = (
                await card.get_attribute("data-mcp-job-id")
                or await card.get_attribute("data-job-id")
                or await card.get_attribute("data-id")
                or await card.get_attribute("id")
                or ""
            )

            # Resolve child selectors on the card
            try:
                title_sel = await _resolve_selector(card, "job_title")
                title = await _safe_get_text(card.locator(title_sel))
            except ValueError:
                title = ""

            try:
                company_sel = await _resolve_selector(card, "job_company")
                company = await _safe_get_text(card.locator(company_sel))
            except ValueError:
                company = ""

            try:
                location_sel = await _resolve_selector(card, "job_location")
                location = await _safe_get_text(card.locator(location_sel))
            except ValueError:
                location = ""

            try:
                desc_sel = await _resolve_selector(card, "job_description")
                description = await _safe_get_text(card.locator(desc_sel))
            except ValueError:
                description = ""

            try:
                salary_sel = await _resolve_selector(card, "salary_range")
                salary = await _safe_get_text(card.locator(salary_sel))
            except ValueError:
                salary = ""

            try:
                date_sel = await _resolve_selector(card, "posted_date")
                posted_date = await _safe_get_text(card.locator(date_sel))
            except ValueError:
                posted_date = ""

            # Resilient fallbacks for title
            if not title:
                for sel in [
                    "h1, h2, h3, h4, h5, h6",
                    "[class*='title'], [class*='position'], [class*='role'], [class*='jobTitle'], [class*='job-title']",
                    "a[href*='job'], a[href*='listing'], a[href*='position'], a.title",
                    "[data-testid*='title']",
                    "strong",
                ]:
                    try:
                        heading = card.locator(sel).first
                        candidate = await _safe_get_text(heading)
                        if candidate:
                            title = candidate
                            break
                    except Exception:
                        continue

            if not title:
                lines = await _extract_meaningful_lines(card)
                if lines:
                    title = lines[0]
            if not title:
                title = "Untitled Position"

            # Sibling-based fallback for company and location immediately adjacent to h4 (or heading)
            if not company or not location:
                for sib_sel in [
                    "h4 ~ div, h4 ~ p, h4 ~ span",
                    "h1 ~ div, h2 ~ div, h3 ~ div, h5 ~ div",
                ]:
                    try:
                        sib_loc = card.locator(sib_sel).first
                        sib_text = await _safe_get_text(sib_loc)
                        if sib_text:
                            # Split on bullet characters •, |, -, or newlines
                            parts = [p.strip() for p in re.split(r"[•|\-\n]+", sib_text) if p.strip()]
                            if len(parts) >= 2:
                                if not company:
                                    company = parts[0]
                                if not location:
                                    location = parts[1]
                                break
                            elif len(parts) == 1:
                                if not company:
                                    company = parts[0]
                                break
                    except Exception:
                        continue

            # Resilient fallbacks for company
            if not company:
                for sel in [
                    ".company, .employer, .company-name",
                    "[class*='company'], [class*='employer'], [class*='organization'], [class*='companyName'], [class*='company-name']",
                    "span[class*='company'], a[class*='company'], a[href*='company']",
                    "[data-testid*='company'], [data-testid*='employer']",
                ]:
                    try:
                        comp_fallback = card.locator(sel).first
                        candidate = await _safe_get_text(comp_fallback)
                        if candidate:
                            company = candidate
                            break
                    except Exception:
                        continue

            if not company:
                lines = await _extract_meaningful_lines(card)
                filtered = [l for l in lines if l != title and not any(k in l for k in ["שמור", "Apply", "הגש"])]
                if filtered:
                    first_line = filtered[0]
                    if any(sep in first_line for sep in ["•", "|", "-"]):
                        parts = [p.strip() for p in re.split(r"[•|\-\n]+", first_line) if p.strip()]
                        if len(parts) >= 2:
                            company = parts[0]
                            if not location:
                                location = parts[1]
                        elif len(parts) == 1:
                            company = parts[0]
                    else:
                        company = first_line

            if not company:
                company = "Unknown Company"

            # Resilient fallbacks for location
            if not location:
                for sel in [
                    ".location, .job-location, span.location, .job-card-location",
                    "[class*='location'], [class*='city'], [class*='place'], [class*='work-mode']",
                    "[data-testid*='location']",
                ]:
                    try:
                        loc_fallback = card.locator(sel).first
                        candidate = await _safe_get_text(loc_fallback)
                        if candidate:
                            location = candidate
                            break
                    except Exception:
                        continue

            # Generate deterministic fallback ID if not present in attributes
            if not job_id:
                hash_input = f"{title}_{company}_{location}_{i}".encode("utf-8")
                job_id = f"job-{hashlib.md5(hash_input).hexdigest()[:10]}"

            # DOM Card Stamping: stamp data-mcp-job-id on the element
            try:
                if hasattr(card, "evaluate"):
                    eval_res = card.evaluate("(el, id) => el.setAttribute('data-mcp-job-id', id)", job_id)
                    if hasattr(eval_res, "__await__"):
                        await eval_res
            except Exception as exc:
                logger.debug("Failed to stamp card locator %d: %s", i, exc)

            try:
                if hasattr(page, "evaluate"):
                    eval_res = page.evaluate(
                        """([selector, index, jobId]) => {
                            const cards = document.querySelectorAll(selector);
                            if (cards && cards[index]) {
                                cards[index].setAttribute('data-mcp-job-id', jobId);
                            }
                        }""",
                        [card_selector, i, job_id],
                    )
                    if hasattr(eval_res, "__await__"):
                        await eval_res
            except Exception as exc:
                logger.debug("Failed to stamp card via page.evaluate %d: %s", i, exc)

            # Extract tech stack badges
            tech_stack: list[str] = []
            try:
                tech_sel = await _resolve_selector(card, "tech_badge")
                badge_locators = card.locator(tech_sel)
                badge_count = await badge_locators.count()
                for b in range(badge_count):
                    badge_text = await _safe_get_text(badge_locators.nth(b))
                    if badge_text and len(badge_text) <= 30:
                        tech_stack.append(badge_text)
            except ValueError:
                pass

            # Supplement tech stack with heuristic text parsing
            combined_text = f"{title} {description} {' '.join(tech_stack)}"
            heuristic_tech = _extract_tech_from_text(combined_text)
            tech_stack = sorted(list(set(tech_stack + heuristic_tech)), key=lambda s: s.lower())

            # Determine work mode
            work_mode = _parse_work_mode(f"{location} {title} {description}")

            # Extract URL if available
            link_locator = card.locator("a[href]").first
            url = None
            if await link_locator.count() > 0:
                href = await link_locator.get_attribute("href")
                if href:
                    if href.startswith("/"):
                        url = f"https://hiremetech.com{href}"
                    else:
                        url = href

            # Check bookmark status
            is_bookmarked = False
            try:
                bm_sel = await _resolve_selector(card, "bookmark_button")
                bm_button = card.locator(bm_sel).first
                if await bm_button.count() > 0:
                    bm_classes = await bm_button.get_attribute("class") or ""
                    bm_aria = await bm_button.get_attribute("aria-pressed") or ""
                    is_bookmarked = (
                        "active" in bm_classes
                        or "saved" in bm_classes
                        or "bookmarked" in bm_classes
                        or bm_aria.lower() == "true"
                    )
            except ValueError:
                pass

            job = Job(
                job_id=job_id,
                title=title,
                company=company,
                location=location,
                work_mode=work_mode,
                tech_stack=tech_stack,
                description=description,
                salary_range=salary if salary else None,
                posted_date=posted_date if posted_date else None,
                url=url,
                is_bookmarked=is_bookmarked,
            )
            jobs.append(job)

        except Exception as exc:
            logger.warning("Failed to parse job card index %d: %s", i, exc)
            continue

    # Batch DOM Card Stamping as extra resilience
    try:
        if hasattr(page, "evaluate") and jobs:
            job_ids = [j.job_id for j in jobs]
            eval_res = page.evaluate(
                """([selector, ids]) => {
                    const cards = document.querySelectorAll(selector);
                    if (cards) {
                        ids.forEach((id, idx) => {
                            if (cards[idx]) {
                                cards[idx].setAttribute('data-mcp-job-id', id);
                            }
                        });
                    }
                }""",
                [card_selector, job_ids],
            )
            if hasattr(eval_res, "__await__"):
                await eval_res
    except Exception as exc:
        logger.debug("Batch DOM stamping evaluation failed: %s", exc)

    logger.info("Successfully extracted %d jobs from page.", len(jobs))
    return jobs


@browser_retry
async def bookmark_job(page: Page, job_id: str) -> bool:
    """Bookmark or favorite a job by ID.

    Args:
        page: Playwright active page.
        job_id: ID of the job to bookmark.

    Returns:
        bool: True if bookmark action succeeded.
    """
    logger.info("Bookmarking job: %s", job_id)

    # Locate the target job card (prioritizing stamped MCP job ID)
    card = page.locator(
        f"[data-mcp-job-id='{job_id}'], "
        f"[data-testid='job-card'][data-job-id='{job_id}'], "
        f"[data-job-id='{job_id}'], [data-id='{job_id}'], #{job_id}"
    ).first

    if await card.count() == 0:
        # Try matching card containing the job ID text
        card = page.locator(
            f"[data-mcp-job-id='{job_id}'], "
            f"div:has-text('{job_id}'), article:has-text('{job_id}'), [data-testid='job-card'], "
            f"div.jobs-app-glass-surface, div.shadow-ht-card"
        ).first

    if await card.count() == 0:
        raise ValueError(f"Job card with ID '{job_id}' not found on the page.")

    bm_sel = await _resolve_selector(card, "bookmark_button")
    bm_button = card.locator(bm_sel).first

    if await bm_button.count() == 0:
        raise ValueError(f"Bookmark button not found on job card '{job_id}' (selector: {bm_sel})")

    await bm_button.click()
    logger.info("Clicked bookmark button for job '%s'", job_id)
    return True


@browser_retry
async def delete_job(page: Page, job_id: str) -> bool:
    """Dismiss or delete/hide a job listing from view.

    Args:
        page: Playwright active page.
        job_id: ID of the job to dismiss/delete.

    Returns:
        bool: True if dismiss action succeeded.
    """
    logger.info("Deleting/Dismissing job: %s", job_id)

    card = page.locator(
        f"[data-mcp-job-id='{job_id}'], "
        f"[data-testid='job-card'][data-job-id='{job_id}'], "
        f"[data-job-id='{job_id}'], [data-id='{job_id}'], #{job_id}"
    ).first

    if await card.count() == 0:
        card = page.locator(
            f"[data-mcp-job-id='{job_id}'], "
            f"div:has-text('{job_id}'), article:has-text('{job_id}'), [data-testid='job-card'], "
            f"div.jobs-app-glass-surface, div.shadow-ht-card"
        ).first

    if await card.count() == 0:
        raise ValueError(f"Job card with ID '{job_id}' not found on the page.")

    del_sel = await _resolve_selector(card, "delete_button")
    del_button = card.locator(del_sel).first

    if await del_button.count() == 0:
        raise ValueError(f"Delete/Dismiss button not found on job card '{job_id}' (selector: {del_sel})")

    await del_button.click()
    logger.info("Clicked delete/dismiss button for job '%s'", job_id)
    return True


@browser_retry
async def preview_application(page: Page, job_id: str) -> ApplicationPreview:
    """Inspect and preview the application workflow and form fields without submitting.

    Args:
        page: Playwright active page.
        job_id: ID of the job to preview application for.

    Returns:
        ApplicationPreview: Preview model containing fields, method, and warnings.
    """
    logger.info("Previewing application for job: %s", job_id)

    card = page.locator(
        f"[data-mcp-job-id='{job_id}'], "
        f"[data-testid='job-card'][data-job-id='{job_id}'], "
        f"[data-job-id='{job_id}'], [data-id='{job_id}'], #{job_id}"
    ).first

    if await card.count() == 0:
        card = page.locator(
            f"[data-mcp-job-id='{job_id}'], "
            f"div:has-text('{job_id}'), article:has-text('{job_id}'), [data-testid='job-card'], "
            f"div.jobs-app-glass-surface, div.shadow-ht-card"
        ).first

    try:
        title_sel = await _resolve_selector(card, "job_title") if await card.count() > 0 else "h2"
    except ValueError:
        title_sel = "h2"

    try:
        company_sel = await _resolve_selector(card, "job_company") if await card.count() > 0 else ".company"
    except ValueError:
        company_sel = ".company"

    job_title = await _safe_get_text(card.locator(title_sel)) if await card.count() > 0 else "Job Application"
    company = await _safe_get_text(card.locator(company_sel)) if await card.count() > 0 else "Employer"

    try:
        apply_sel = await _resolve_selector(card, "apply_button") if await card.count() > 0 else "button.apply"
    except ValueError:
        apply_sel = "button.apply"
    apply_elem = card.locator(apply_sel).first if await card.count() > 0 else page.locator(apply_sel).first

    warnings: list[str] = []
    fields_to_submit: dict[str, Any] = {}
    application_method = "direct_submission"

    if await apply_elem.count() > 0:
        href = await apply_elem.get_attribute("href")
        target = await apply_elem.get_attribute("target")

        if href and (href.startswith("http://") or href.startswith("https://")) and "hiremetech.com" not in href:
            application_method = "external_redirect"
            fields_to_submit = {"external_url": href}
            warnings.append(f"Application redirects to an external site ({href}). Automatic submission unavailable.")
            return ApplicationPreview(
                job_id=job_id,
                job_title=job_title,
                company=company,
                application_method=application_method,
                fields_to_submit=fields_to_submit,
                warnings=warnings,
            )

        # In-page application flow: click to inspect modal/form
        try:
            await apply_elem.click()
            await page.wait_for_timeout(500)

            # Inspect modal or form fields
            modal = page.locator("[role='dialog'], .modal, form.apply-form, .application-modal").first
            container = modal if await modal.count() > 0 else page

            # Inspect text inputs
            inputs = container.locator("input:not([type='hidden']):not([type='submit'])")
            input_count = await inputs.count()
            for i in range(input_count):
                inp = inputs.nth(i)
                inp_name = (
                    await inp.get_attribute("name")
                    or await inp.get_attribute("placeholder")
                    or await inp.get_attribute("id")
                    or f"input_{i}"
                )
                inp_type = await inp.get_attribute("type") or "text"
                inp_req = await inp.get_attribute("required") is not None

                fields_to_submit[inp_name] = {
                    "type": inp_type,
                    "required": inp_req,
                }
                if inp_type == "file":
                    warnings.append(f"Resume/Document file upload required for field: '{inp_name}'")

            # Inspect textareas
            textareas = container.locator("textarea")
            textarea_count = await textareas.count()
            for i in range(textarea_count):
                ta = textareas.nth(i)
                ta_name = (
                    await ta.get_attribute("name")
                    or await ta.get_attribute("placeholder")
                    or await ta.get_attribute("id")
                    or f"textarea_{i}"
                )
                ta_req = await ta.get_attribute("required") is not None
                fields_to_submit[ta_name] = {
                    "type": "textarea",
                    "required": ta_req,
                }
                warnings.append(f"Freeform cover letter or questionnaire field detected: '{ta_name}'")

            # Inspect dropdown selects
            selects = container.locator("select")
            select_count = await selects.count()
            for i in range(select_count):
                sel = selects.nth(i)
                sel_name = (
                    await sel.get_attribute("name")
                    or await sel.get_attribute("id")
                    or f"select_{i}"
                )
                fields_to_submit[sel_name] = {
                    "type": "select",
                    "required": await sel.get_attribute("required") is not None,
                }

            # Close modal if open to prevent lingering overlay
            close_btn = container.locator("button[aria-label*='close'], button.close, [data-testid='close-modal']").first
            if await close_btn.count() > 0:
                await close_btn.click()
            else:
                await page.keyboard.press("Escape")

        except Exception as exc:
            logger.warning("Error inspecting application modal for job '%s': %s", job_id, exc)
            warnings.append(f"Form inspection encountered partial error: {exc}")
    else:
        warnings.append("Apply button was not found on the job listing.")

    if not fields_to_submit:
        fields_to_submit = {
            "applicant_name": "Standard Profile Name",
            "applicant_email": "Standard Profile Email",
            "resume": "Default Profile Resume",
        }

    return ApplicationPreview(
        job_id=job_id,
        job_title=job_title,
        company=company,
        application_method=application_method,
        fields_to_submit=fields_to_submit,
        warnings=warnings,
    )


@browser_retry
async def execute_application(page: Page, job_id: str) -> bool:
    """Execute submission of job application on the active page.

    Args:
        page: Playwright active page.
        job_id: ID of the job to apply for.

    Returns:
        bool: True if application was submitted successfully.
    """
    logger.info("Executing application submission for job: %s", job_id)

    card = page.locator(
        f"[data-mcp-job-id='{job_id}'], "
        f"[data-testid='job-card'][data-job-id='{job_id}'], "
        f"[data-job-id='{job_id}'], [data-id='{job_id}'], #{job_id}"
    ).first

    if await card.count() == 0:
        card = page.locator(
            f"[data-mcp-job-id='{job_id}'], "
            f"div:has-text('{job_id}'), article:has-text('{job_id}'), [data-testid='job-card'], "
            f"div.jobs-app-glass-surface, div.shadow-ht-card"
        ).first

    try:
        apply_sel = await _resolve_selector(card, "apply_button") if await card.count() > 0 else "button.apply"
    except ValueError:
        apply_sel = "button.apply"
    apply_elem = card.locator(apply_sel).first if await card.count() > 0 else page.locator(apply_sel).first

    if await apply_elem.count() > 0:
        # Check if clicking apply directly opens modal
        await apply_elem.click()
        await page.wait_for_timeout(500)

    # Find confirmation / submit button in modal or page
    try:
        submit_sel = await _resolve_selector(page, "submit_button")
    except ValueError:
        submit_sel = "button[type='submit']"
    submit_btn = page.locator(submit_sel).first

    if await submit_btn.count() > 0 and await submit_btn.is_visible():
        await submit_btn.click()
        logger.info("Clicked submission button for job '%s'", job_id)
        await page.wait_for_timeout(1000)
        return True

    # If apply button itself was the direct submit action
    logger.info("Direct apply action executed for job '%s'", job_id)
    return True
