"""Adaptive Heuristic DOM Discovery and Dynamic Selector Mapping for Tech Job  MCP."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import Locator, Page

from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DYNAMIC_SELECTORS_PATH = os.path.expanduser("~/.hireme_mcp/dynamic_selectors.json")

# Default selector definitions used across the browser automation layer
SELECTORS: dict[str, dict[str, str]] = {
    "job_card": {
        "primary": "div.jobs-app-glass-surface, div.shadow-ht-card, [data-testid='job-card']",
        "fallback": "div[class*='jobs-app-glass-surface'], div[class*='shadow-ht-card'], .job-listing-card, .job-item, article.job, div.job-card",
    },
    "job_title": {
        "primary": "h4.font-bold, h4[class*='text-gray-900'], [data-testid='job-title']",
        "fallback": "h4, h2.job-title, h3.title, .job-card-title, a.job-title, [class*='title'], [class*='position'], [class*='role'], [class*='jobTitle'], [class*='job-title']",
    },
    "job_company": {
        "primary": "[data-testid='company-name']",
        "fallback": ".company-name, .employer, span.company, a.company, [class*='company'], [class*='employer'], [class*='organization'], [class*='companyName'], [class*='company-name']",
    },
    "job_location": {
        "primary": "[data-testid='job-location']",
        "fallback": ".job-location, .location, span.location, .job-card-location",
    },
    "bookmark_button": {
        "primary": "button.bg-ht-primary-500, button[class*='bg-ht-primary'], [data-testid='bookmark-btn']",
        "fallback": "button[aria-label*='save' i], button[aria-label*='bookmark' i], button:has-text('שמור'), button:has-text('שמירה'), button[aria-label*='שמור' i], button[aria-label*='שמירה' i], button.bookmark, .favorite-btn",
    },
    "delete_button": {
        "primary": "[data-testid='delete-btn']",
        "fallback": "button.dismiss, button[aria-label*='hide' i], button:has-text('מחק'), button:has-text('הסר'), button:has-text('הסרה'), button[aria-label*='מחק' i], button[aria-label*='הסר' i], button[aria-label*='הסרה' i], .remove-job-btn",
    },
    "apply_button": {
        "primary": "[data-testid='apply-btn']",
        "fallback": "button.apply, a.apply-now, .auto-apply-btn, button:has-text('Apply'), button:has-text('הגש'), button:has-text('הגשת מועמדות'), button:has-text('הגש מועמדות'), a:has-text('הגש'), a:has-text('הגשת מועמדות'), button[aria-label*='הגש' i]",
    },
    "tech_badge": {
        "primary": "[data-testid='tech-badge']",
        "fallback": ".tech-badge, .badge, .tag, .skill-tag, .tech-stack span, span[class*='badge'], span[class*='tag']",
    },
    "job_description": {
        "primary": "[data-testid='job-description']",
        "fallback": ".job-description, .description, .job-details, div[class*='description'], p",
    },
    "salary_range": {
        "primary": "[data-testid='salary-range']",
        "fallback": ".salary, .salary-range, [data-testid='salary'], span[class*='salary']",
    },
    "posted_date": {
        "primary": "[data-testid='posted-date']",
        "fallback": ".posted-date, .date, time, span[class*='date']",
    },
    "submit_button": {
        "primary": "[data-testid='submit-btn']",
        "fallback": "button[type='submit'], button:has-text('Submit'), button:has-text('Send Application'), button:has-text('Confirm'), button:has-text('שלח'), button:has-text('שליחה'), button:has-text('אישור'), button[aria-label*='שלח' i]",
    },
}

# Semantic candidate patterns for heuristic child element discovery
CHILD_ROLE_CANDIDATES: dict[str, list[str]] = {
    "job_card": [
        "div.jobs-app-glass-surface",
        "div.shadow-ht-card",
        ".jobs-app-glass-surface",
        ".shadow-ht-card",
        "[class*='jobs-app-glass-surface']",
        "[class*='shadow-ht-card']",
        "[data-testid='job-card']",
        "[data-testid*='job-card']",
        ".job-listing-card",
        ".job-item",
        "article.job",
        "div.job-card",
    ],
    "job_title": [
        "h4.font-bold",
        "h4[class*='text-gray-900']",
        "[data-testid*='title']",
        "[data-testid*='position']",
        "[data-testid*='role']",
        "h4",
        "h1, h2, h3, h4",
        "[class*='title']",
        "[class*='position']",
        "[class*='role']",
        "[class*='jobTitle']",
        "[class*='job-title']",
        "a[href*='job']",
        "a[href*='listing']",
        "a.title",
        ".job-title",
    ],
    "job_company": [
        "[data-testid*='company']",
        "[data-testid*='employer']",
        "[class*='company']",
        "[class*='employer']",
        "[class*='organization']",
        "[class*='companyName']",
        "[class*='company-name']",
        "span.company",
        "a.company",
        ".company-name",
    ],
    "job_location": [
        "[data-testid*='location']",
        "[class*='location']",
        "[class*='city']",
        "[class*='place']",
        "[class*='work-mode']",
        "span.location",
        ".job-location",
    ],
    "bookmark_button": [
        "button.bg-ht-primary-500",
        "button[class*='bg-ht-primary']",
        "[data-testid*='bookmark']",
        "[data-testid*='save']",
        "[data-testid*='favorite']",
        "button[aria-label*='bookmark' i]",
        "button[aria-label*='save' i]",
        "button[aria-label*='favorite' i]",
        "button[aria-label*='שמור' i]",
        "button[aria-label*='שמירה' i]",
        "button:has-text('שמור')",
        "button:has-text('שמירה')",
        "a:has-text('שמור')",
        "a:has-text('שמירה')",
        "button.bookmark",
        "button.save",
        ".bookmark-btn",
        ".favorite-btn",
        "button:has(svg[class*='bookmark'])",
        "button:has(svg[class*='star'])",
    ],
    "delete_button": [
        "[data-testid*='delete']",
        "[data-testid*='dismiss']",
        "[data-testid*='hide']",
        "[data-testid*='remove']",
        "button[aria-label*='delete' i]",
        "button[aria-label*='dismiss' i]",
        "button[aria-label*='hide' i]",
        "button[aria-label*='remove' i]",
        "button[aria-label*='מחק' i]",
        "button[aria-label*='הסר' i]",
        "button[aria-label*='הסרה' i]",
        "button:has-text('מחק')",
        "button:has-text('הסר')",
        "button:has-text('הסרה')",
        "button.dismiss",
        "button.delete",
        ".dismiss-btn",
        ".remove-job-btn",
        "button:has(svg[class*='trash'])",
        "button:has(svg[class*='close'])",
    ],
    "apply_button": [
        "[data-testid*='apply']",
        "button:has-text('Apply')",
        "a:has-text('Apply')",
        "button:has-text('הגש')",
        "button:has-text('הגשת מועמדות')",
        "button:has-text('הגש מועמדות')",
        "a:has-text('הגש')",
        "a:has-text('הגשת מועמדות')",
        "a:has-text('הגש מועמדות')",
        "button[aria-label*='apply' i]",
        "button[aria-label*='הגש' i]",
        "button[aria-label*='הגשת מועמדות' i]",
        "a[aria-label*='apply' i]",
        "button.apply",
        "a.apply-now",
        ".auto-apply-btn",
        "button:has-text('Quick Apply')",
    ],
    "tech_badge": [
        "[data-testid*='tech']",
        "[data-testid*='badge']",
        "[data-testid*='tag']",
        "[data-testid*='skill']",
        ".tech-badge",
        ".badge",
        ".tag",
        ".skill-tag",
        ".pill",
        ".chip",
        "[class*='badge']",
        "[class*='tag']",
        "[class*='skill']",
        ".tech-stack span",
    ],
    "job_description": [
        "[data-testid*='description']",
        "[data-testid*='summary']",
        ".job-description",
        ".description",
        ".job-details",
        ".summary",
        "div[class*='description']",
        "p",
    ],
    "salary_range": [
        "[data-testid*='salary']",
        "[data-testid*='compensation']",
        ".salary",
        ".salary-range",
        ".compensation",
        "[class*='salary']",
        "span[class*='salary']",
    ],
    "posted_date": [
        "[data-testid*='date']",
        "[data-testid*='posted']",
        "time",
        ".posted-date",
        ".date",
        "[class*='posted']",
        "span[class*='date']",
        "span[class*='time']",
    ],
    "submit_button": [
        "[data-testid*='submit']",
        "button[type='submit']",
        "button:has-text('Submit')",
        "button:has-text('Send Application')",
        "button:has-text('Confirm')",
        "button:has-text('Apply Now')",
        "button:has-text('שלח')",
        "button:has-text('שליחה')",
        "button:has-text('אישור')",
        "button[aria-label*='שלח' i]",
    ],
}

# JavaScript function for DOM sibling clustering analysis
JS_DISCOVER_CARD = """() => {
    function hasJobContent(el) {
        if (!el || el.nodeType !== 1) return false;
        const text = (el.innerText || el.textContent || "").trim();
        if (text.length < 20) return false;
        const hasInteractiveOrHeading = el.querySelector("h1, h2, h3, h4, h5, h6, a, button, [role='button']") !== null;
        return hasInteractiveOrHeading;
    }

    // 1. Data-testid heuristics
    const testIdCandidates = [
        "[data-testid='job-card']",
        "[data-testid*='job-card']",
        "[data-testid*='job-item']",
        "[data-testid*='job-listing']",
        "[data-testid*='listing-card']",
        "[data-testid*='job']",
        "[data-testid*='card']"
    ];
    for (const sel of testIdCandidates) {
        try {
            const matches = document.querySelectorAll(sel);
            if (matches.length >= 3) {
                return sel;
            }
        } catch (e) {}
    }

    // 2. Class-based standard heuristics
    const standardClassSelectors = [
        ".jobs-app-glass-surface", ".shadow-ht-card",
        ".job-listing-card", ".job-item", "article.job", "div.job-card",
        ".job-card", ".job-listing", ".card-job", "[class*='job-card']",
        "[class*='JobCard']", "[class*='jobCard']", "[class*='job-listing']",
        "[class*='job_card']"
    ];
    for (const sel of standardClassSelectors) {
        try {
            const matches = document.querySelectorAll(sel);
            if (matches.length >= 3) {
                return sel;
            }
        } catch (e) {}
    }

    // 3. Structural Sibling Clustering in Main/Body
    const rootContainers = Array.from(document.querySelectorAll("main, [role='main'], #root, #__next, #app, body"));
    for (const root of rootContainers) {
        const potentialParents = Array.from(root.querySelectorAll("*"));
        potentialParents.unshift(root);

        for (const parent of potentialParents) {
            const children = Array.from(parent.children);
            if (children.length < 3) continue;

            const jobLikeChildren = children.filter(hasJobContent);
            if (jobLikeChildren.length >= 3) {
                const classCounts = {};
                for (const child of jobLikeChildren) {
                    for (const cls of Array.from(child.classList)) {
                        if (cls.trim().length > 0) {
                            classCounts[cls] = (classCounts[cls] || 0) + 1;
                        }
                    }
                }

                let bestClass = null;
                let maxCount = 0;
                for (const [cls, count] of Object.entries(classCounts)) {
                    if (count >= 3 && count > maxCount) {
                        bestClass = cls;
                        maxCount = count;
                    }
                }

                if (bestClass) {
                    const tag = jobLikeChildren[0].tagName.toLowerCase();
                    const tagClassCandidate = `${tag}.${bestClass}`;
                    if (document.querySelectorAll(tagClassCandidate).length >= 3) {
                        return tagClassCandidate;
                    }
                    const classCandidate = `.${bestClass}`;
                    if (document.querySelectorAll(classCandidate).length >= 3) {
                        return classCandidate;
                    }
                }

                const tagCounts = {};
                for (const child of jobLikeChildren) {
                    const tag = child.tagName.toLowerCase();
                    tagCounts[tag] = (tagCounts[tag] || 0) + 1;
                }
                for (const [tag, count] of Object.entries(tagCounts)) {
                    if (count >= 3 && (tag === "article" || tag === "li" || tag === "section")) {
                        if (document.querySelectorAll(tag).length >= 3) {
                            return tag;
                        }
                    }
                }
            }
        }
    }

    return null;
}"""


class DynamicSelectorRegistry:
    """Persistent storage for discovered working DOM selectors."""

    def __init__(self, file_path: Optional[str | Path] = None) -> None:
        if file_path is not None:
            self.file_path = Path(file_path).expanduser().resolve()
        else:
            env_path = os.environ.get("DYNAMIC_SELECTORS_PATH")
            if env_path:
                self.file_path = Path(env_path).expanduser().resolve()
            else:
                self.file_path = Path(DEFAULT_DYNAMIC_SELECTORS_PATH).expanduser().resolve()

        self._selectors: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        """Load selectors from the JSON file if available."""
        if not self.file_path.exists():
            self._selectors = {}
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._selectors = {str(k): str(v) for k, v in data.items()}
                else:
                    logger.warning("Selectors file '%s' format invalid; initializing empty.", self.file_path)
                    self._selectors = {}
        except Exception as exc:
            logger.warning("Failed to load selectors from '%s': %s", self.file_path, exc)
            self._selectors = {}

    def save(self) -> None:
        """Persist selectors dictionary to disk."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._selectors, f, indent=2)
            logger.debug("Saved dynamic selectors to '%s'", self.file_path)
        except Exception as exc:
            logger.error("Failed to save dynamic selectors to '%s': %s", self.file_path, exc)

    def get(self, key: str) -> Optional[str]:
        """Retrieve stored selector by key."""
        return self._selectors.get(key)

    def set(self, key: str, selector: str) -> None:
        """Store selector by key and persist to disk."""
        self._selectors[key] = selector
        self.save()

    def get_all(self) -> dict[str, str]:
        """Return a copy of all registered dynamic selectors."""
        return self._selectors.copy()

    def clear(self) -> None:
        """Clear all stored selectors and update persistence file."""
        self._selectors.clear()
        self.save()


async def discover_card_selector(page: Page) -> Optional[str]:
    """Execute DOM analysis to discover job card container selector based on sibling clustering.

    Args:
        page: Active Playwright page.

    Returns:
        Optional[str]: Discovered CSS selector matching repeating job cards, or None.
    """
    try:
        if not hasattr(page, "evaluate"):
            return None

        eval_res = page.evaluate(JS_DISCOVER_CARD)
        candidate = await eval_res if hasattr(eval_res, "__await__") else eval_res

        if candidate and isinstance(candidate, str):
            count = await page.locator(candidate).count()
            if count > 0:
                logger.info("Heuristically discovered card selector: '%s' (%d elements)", candidate, count)
                return candidate
    except Exception as exc:
        logger.warning("Error during discover_card_selector: %s", exc)
    return None


async def discover_child_selector(
    card_locator_or_page: Page | Locator,
    role: str,
) -> Optional[str]:
    """Perform scoped semantic search for a child element inside a job card.

    Args:
        card_locator_or_page: Card locator or page to search within.
        role: Child element role identifier (e.g. 'job_title', 'apply_button').

    Returns:
        Optional[str]: Matching selector if found, else None.
    """
    candidates = CHILD_ROLE_CANDIDATES.get(role, [])
    for candidate in candidates:
        try:
            loc = card_locator_or_page.locator(candidate)
            if await loc.count() > 0:
                logger.info("Heuristically discovered child selector for '%s': '%s'", role, candidate)
                return candidate
        except Exception:
            continue
    return None


async def calibrate_all_selectors(
    page: Page,
    registry: Optional[DynamicSelectorRegistry] = None,
    selectors_map: Optional[dict[str, dict[str, str]]] = None,
) -> dict[str, dict[str, Any]]:
    """Test all registered selectors against the live page DOM and record calibration results.

    Args:
        page: Active Playwright page.
        registry: Optional DynamicSelectorRegistry to persist verified selectors.
        selectors_map: Selector mapping dictionary (defaults to SELECTORS).

    Returns:
        dict[str, dict[str, Any]]: Calibration report per selector key with status, active selector, and match count.
    """
    if selectors_map is None:
        selectors_map = SELECTORS

    results: dict[str, dict[str, Any]] = {}

    # 1. First calibrate job_card container
    card_key = "job_card"
    card_info = selectors_map.get(card_key, {})
    primary_card = card_info.get("primary", "")
    fallback_card = card_info.get("fallback", "")

    card_status = "failed"
    card_active_sel: Optional[str] = None
    card_count = 0

    # Try primary
    if primary_card:
        try:
            cnt = await page.locator(primary_card).count()
            if cnt > 0:
                card_status = "primary_matched"
                card_active_sel = primary_card
                card_count = cnt
        except Exception:
            pass

    # Try fallback if primary didn't match
    if card_status == "failed" and fallback_card:
        for sub_fallback in [f.strip() for f in fallback_card.split(",")]:
            try:
                cnt = await page.locator(sub_fallback).count()
                if cnt > 0:
                    card_status = "fallback_matched"
                    card_active_sel = sub_fallback
                    card_count = cnt
                    break
            except Exception:
                continue

    # Try heuristic discovery if fallback didn't match
    if card_status == "failed":
        try:
            discovered = await discover_card_selector(page)
            if discovered:
                cnt = await page.locator(discovered).count()
                if cnt > 0:
                    card_status = "discovered_heuristic"
                    card_active_sel = discovered
                    card_count = cnt
        except Exception:
            pass

    results[card_key] = {
        "status": card_status,
        "selector": card_active_sel,
        "count": card_count,
    }

    if card_active_sel and registry is not None:
        registry.set(card_key, card_active_sel)

    # 2. Get card context locator if card container was found
    if card_active_sel and card_count > 0:
        card_context: Page | Locator = page.locator(card_active_sel).first
    else:
        card_context = page

    # 3. Calibrate all other child / form selectors
    for key, info in selectors_map.items():
        if key == card_key:
            continue

        primary = info.get("primary", "")
        fallback = info.get("fallback", "")

        status = "failed"
        active_sel: Optional[str] = None
        count = 0

        # Try primary
        if primary:
            try:
                cnt = await card_context.locator(primary).count()
                if cnt > 0:
                    status = "primary_matched"
                    active_sel = primary
                    count = cnt
            except Exception:
                pass

        # Try fallback
        if status == "failed" and fallback:
            for sub_fallback in [f.strip() for f in fallback.split(",")]:
                try:
                    cnt = await card_context.locator(sub_fallback).count()
                    if cnt > 0:
                        status = "fallback_matched"
                        active_sel = sub_fallback
                        count = cnt
                        break
                except Exception:
                    continue

        # Try heuristic discovery
        if status == "failed":
            try:
                discovered = await discover_child_selector(card_context, key)
                if discovered:
                    cnt = await card_context.locator(discovered).count()
                    if cnt > 0:
                        status = "discovered_heuristic"
                        active_sel = discovered
                        count = cnt
            except Exception:
                pass

        results[key] = {
            "status": status,
            "selector": active_sel,
            "count": count,
        }

        if active_sel and registry is not None:
            registry.set(key, active_sel)

    return results
