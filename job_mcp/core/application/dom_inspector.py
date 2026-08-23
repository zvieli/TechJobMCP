"""Universal DOM Inspector and Form Schema Extractor for Playwright.

Provides zero-guesswork, research-backed DOM form extraction across main document
and all frames/iframes dynamically. Extracts form fields, associated labels,
file uploads, selects, radios, checkboxes, and deterministically identifies
application submit buttons with optional LLM disambiguation.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, List, Optional, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FormFieldSchema(BaseModel):
    """Schema definition for an extracted DOM form field."""

    field_id: str = Field(description="Unique DOM ID, name, or generated selector for the field")
    name: str = Field(default="", description="HTML name attribute or identifier")
    label: str = Field(default="", description="Associated label, aria-label, placeholder, or contextual text")
    field_type: str = Field(
        default="text",
        description="Field type (text, email, tel, file, select, textarea, radio, checkbox, number, url, etc.)",
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="List of options for select, radio, or multi-choice controls",
    )
    frame_index: int = Field(default=0, description="Index of frame in page.frames containing this element")
    required: bool = Field(default=False, description="Whether the field is marked required")
    placeholder: Optional[str] = Field(default=None, description="Input placeholder text if present")
    value: Optional[str] = Field(default=None, description="Current field value if any")
    selector: Optional[str] = Field(default=None, description="Playwright CSS selector for direct interaction")

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary representation."""
        return self.model_dump()


class SubmitButtonInfo(BaseModel):
    """Information regarding a detected submit or continue button."""

    frame_index: int = Field(default=0, description="Index of frame containing the submit button")
    selector: str = Field(description="CSS selector or locator string for the button")
    text: str = Field(default="", description="Visible text or aria-label of the button")
    button_type: str = Field(default="submit", description="HTML button type or role")
    confidence: float = Field(default=1.0, description="Heuristic or LLM confidence score (0.0 - 1.0)")
    action_type: str = Field(default="submit", description="Inferred action: 'submit', 'continue', 'next', 'review'")
    element_id: Optional[str] = Field(default=None, description="HTML id attribute if available")
    element_class: Optional[str] = Field(default=None, description="HTML class attribute if available")

    def get_locator(self, page: Any) -> Any:
        """Resolve Playwright Locator for this submit button on the given page.

        Args:
            page: Playwright Page instance.

        Returns:
            Locator instance targeted at the submit button in the appropriate frame.
        """
        if hasattr(page, "frames") and 0 <= self.frame_index < len(page.frames):
            target = page.frames[self.frame_index]
        else:
            target = page
        return target.locator(self.selector)


# ---------------------------------------------------------------------------
# JavaScript In-Browser Extraction Script
# ---------------------------------------------------------------------------

_DOM_EXTRACTOR_JS = """
() => {
    function cleanText(str) {
        if (!str) return '';
        return str.replace(/\\s+/g, ' ').trim();
    }

    function isElementVisible(el) {
        if (!el) return false;
        if (el.type === 'file') return true;
        if (el.type === 'hidden') return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
            return false;
        }
        const rect = el.getBoundingClientRect();
        return rect.width > 0 || rect.height > 0 || el.getClientRects().length > 0;
    }

    function getElementLabel(el) {
        // 1. aria-label
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel && cleanText(ariaLabel)) {
            return cleanText(ariaLabel);
        }

        // 2. aria-labelledby
        const ariaLabelledBy = el.getAttribute('aria-labelledby');
        if (ariaLabelledBy) {
            const labelEl = document.getElementById(ariaLabelledBy);
            if (labelEl) {
                const txt = cleanText(labelEl.innerText || labelEl.textContent);
                if (txt) return txt;
            }
        }

        // 3. HTML5 labels list property
        if (el.labels && el.labels.length > 0) {
            const lblText = Array.from(el.labels).map(l => cleanText(l.innerText || l.textContent)).filter(Boolean).join(' ');
            if (lblText) return lblText;
        }

        // 4. <label for="id">
        if (el.id) {
            try {
                const forLabel = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
                if (forLabel) {
                    const txt = cleanText(forLabel.innerText || forLabel.textContent);
                    if (txt) return txt;
                }
            } catch (e) {}
        }

        // 5. Parent / wrapping <label>
        const parentLabel = el.closest('label');
        if (parentLabel) {
            const clone = parentLabel.cloneNode(true);
            const nested = clone.querySelectorAll('input, select, textarea');
            nested.forEach(n => n.remove());
            const txt = cleanText(clone.innerText || clone.textContent);
            if (txt) return txt;
        }

        // 6. Immediate preceding sibling label or text (stopping at other input controls)
        let prev = el.previousElementSibling;
        if (prev) {
            const tag = prev.tagName.toUpperCase();
            if (['LABEL', 'SPAN', 'P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LEGEND'].includes(tag)) {
                const txt = cleanText(prev.innerText || prev.textContent);
                if (txt && txt.length < 120) return txt;
            }
        }

        // 7. Fieldset legend
        const fieldset = el.closest('fieldset');
        if (fieldset) {
            const legend = fieldset.querySelector('legend');
            if (legend) {
                const txt = cleanText(legend.innerText || legend.textContent);
                if (txt) return txt;
            }
        }

        // 8. Form-group container header
        const formGroup = el.closest('.form-group, .field, .form-field, .input-group');
        if (formGroup) {
            const header = formGroup.querySelector('label, .label, .title, .field-label');
            if (header && header !== el && !header.contains(el)) {
                const txt = cleanText(header.innerText || header.textContent);
                if (txt && txt.length < 120) return txt;
            }
        }

        // 9. Placeholder
        const placeholder = el.getAttribute('placeholder');
        if (placeholder && cleanText(placeholder)) {
            return cleanText(placeholder);
        }

        // 10. Title attribute
        const title = el.getAttribute('title');
        if (title && cleanText(title)) {
            return cleanText(title);
        }

        // 11. Fallback to name or id
        return el.name || el.id || '';
    }

    function isElementRequired(el) {
        if (el.required) return true;
        if (el.getAttribute('aria-required') === 'true') return true;
        if (el.getAttribute('required') !== null) return true;
        const parent = el.closest('.form-group, .field, .form-field, label, div');
        if (parent) {
            if (parent.classList.contains('required') || parent.getAttribute('data-required') === 'true') return true;
            const text = parent.innerText || '';
            if (text.includes('*') || /\\b(required|mandatory)\\b/i.test(text)) return true;
        }
        return false;
    }

    function generateSelector(el, index) {
        if (el.id) {
            try {
                return `#${CSS.escape(el.id)}`;
            } catch (e) {}
        }
        const testId = el.getAttribute('data-testid') || el.getAttribute('data-qa') || el.getAttribute('data-automation-id');
        if (testId) {
            return `[data-testid="${testId}"], [data-qa="${testId}"], [data-automation-id="${testId}"]`;
        }
        if (el.name) {
            try {
                const tag = el.tagName.toLowerCase();
                const type = el.getAttribute('type');
                if (type) {
                    return `${tag}[name="${CSS.escape(el.name)}"][type="${type}"]`;
                }
                return `${tag}[name="${CSS.escape(el.name)}"]`;
            } catch (e) {}
        }
        const tag = el.tagName.toLowerCase();
        return `${tag}:nth-of-type(${index + 1})`;
    }

    const rawElements = [];
    const radioGroups = {};

    // Collect inputs
    const inputs = document.querySelectorAll('input:not([type="hidden"]):not([type="submit"]):not([type="button"]):not([type="reset"]):not([type="image"])');
    inputs.forEach((el, idx) => {
        if (!isElementVisible(el)) return;
        const type = (el.getAttribute('type') || 'text').toLowerCase();
        const name = el.getAttribute('name') || '';
        const id = el.id || '';
        const label = getElementLabel(el);
        const req = isElementRequired(el);
        const placeholder = el.getAttribute('placeholder') || null;
        const value = el.value || null;
        const selector = generateSelector(el, idx);

        if (type === 'radio') {
            const groupKey = name || id || `radio_group_${idx}`;
            if (!radioGroups[groupKey]) {
                radioGroups[groupKey] = {
                    field_id: id || `radio_${groupKey}`,
                    name: name,
                    label: label,
                    field_type: 'radio',
                    options: [],
                    required: req,
                    selector: name ? `input[type="radio"][name="${CSS.escape(name)}"]` : selector,
                    frame_index: 0
                };
            }
            const optVal = el.value || label || `option_${radioGroups[groupKey].options.length + 1}`;
            const optLabel = label || optVal;
            if (!radioGroups[groupKey].options.includes(optLabel)) {
                radioGroups[groupKey].options.push(optLabel);
            }
            if (req) radioGroups[groupKey].required = true;
        } else {
            rawElements.push({
                field_id: id || name || `input_${type}_${idx}`,
                name: name,
                label: label,
                field_type: type,
                options: null,
                required: req,
                placeholder: placeholder,
                value: value,
                selector: selector,
                frame_index: 0
            });
        }
    });

    // Add grouped radios
    Object.values(radioGroups).forEach(group => {
        rawElements.push(group);
    });

    // Collect textareas
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach((el, idx) => {
        if (!isElementVisible(el)) return;
        const name = el.getAttribute('name') || '';
        const id = el.id || '';
        const label = getElementLabel(el);
        const req = isElementRequired(el);
        const placeholder = el.getAttribute('placeholder') || null;
        const value = el.value || null;
        const selector = generateSelector(el, idx);

        rawElements.push({
            field_id: id || name || `textarea_${idx}`,
            name: name,
            label: label,
            field_type: 'textarea',
            options: null,
            required: req,
            placeholder: placeholder,
            value: value,
            selector: selector,
            frame_index: 0
        });
    });

    // Collect selects
    const selects = document.querySelectorAll('select');
    selects.forEach((el, idx) => {
        if (!isElementVisible(el)) return;
        const name = el.getAttribute('name') || '';
        const id = el.id || '';
        const label = getElementLabel(el);
        const req = isElementRequired(el);
        const selector = generateSelector(el, idx);

        const options = Array.from(el.options)
            .map(opt => cleanText(opt.innerText || opt.text || opt.value))
            .filter(opt => opt && !/^(select|choose|please select|--)/i.test(opt));

        rawElements.push({
            field_id: id || name || `select_${idx}`,
            name: name,
            label: label,
            field_type: 'select',
            options: options.length > 0 ? options : null,
            required: req,
            placeholder: null,
            value: el.value || null,
            selector: selector,
            frame_index: 0
        });
    });

    return rawElements;
}
"""


_BUTTON_EXTRACTOR_JS = """
() => {
    function cleanText(str) {
        if (!str) return '';
        return str.replace(/\\s+/g, ' ').trim();
    }

    function isElementVisible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
            return false;
        }
        const rect = el.getBoundingClientRect();
        return rect.width > 0 || rect.height > 0 || el.getClientRects().length > 0;
    }

    function generateButtonSelector(el, index) {
        if (el.id) {
            try {
                return `#${CSS.escape(el.id)}`;
            } catch (e) {}
        }
        const testId = el.getAttribute('data-testid') || el.getAttribute('data-qa') || el.getAttribute('data-automation-id');
        if (testId) {
            return `[data-testid="${testId}"], [data-qa="${testId}"], [data-automation-id="${testId}"]`;
        }
        if (el.name) {
            try {
                return `${el.tagName.toLowerCase()}[name="${CSS.escape(el.name)}"]`;
            } catch (e) {}
        }
        const tag = el.tagName.toLowerCase();
        const type = el.getAttribute('type');
        if (type) {
            return `${tag}[type="${type}"]:nth-of-type(${index + 1})`;
        }
        return `${tag}:nth-of-type(${index + 1})`;
    }

    const candidates = [];
    const elements = document.querySelectorAll(
        'button:not([disabled]), input[type="submit"]:not([disabled]), input[type="button"]:not([disabled]), a[role="button"], [role="button"]:not([aria-disabled="true"]), [type="submit"]:not([disabled])'
    );

    elements.forEach((el, idx) => {
        if (!isElementVisible(el)) return;
        const text = cleanText(
            el.innerText || el.textContent || el.getAttribute('value') || el.getAttribute('aria-label') || el.getAttribute('title') || ''
        );
        const type = el.getAttribute('type') || (el.tagName.toLowerCase() === 'button' ? 'button' : el.getAttribute('role') || 'button');
        const selector = generateButtonSelector(el, idx);
        const elId = el.id || null;
        const elClass = el.className || null;

        candidates.push({
            selector: selector,
            text: text,
            button_type: type,
            element_id: elId,
            element_class: elClass,
            frame_index: 0
        });
    });

    return candidates;
}
"""


# ---------------------------------------------------------------------------
# Deterministic Button Scoring Patterns
# ---------------------------------------------------------------------------

_EXACT_SUBMIT_PATTERNS = re.compile(
    r"^(submit(\s+application)?|apply(\s+now)?|send(\s+application)?|complete\s+application|submit\s+resume|finish(\s+application)?|confirm\s+application|confirm|send)$",
    re.IGNORECASE,
)

_CONTAINS_SUBMIT_PATTERNS = re.compile(
    r"\b(submit|apply|send\s+application|complete\s+application|submit\s+resume)\b",
    re.IGNORECASE,
)

_NEXT_STEP_PATTERNS = re.compile(
    r"^(next(\s+step)?|continue|save\s+(&|and)\s+continue|proceed|review(\s+application)?|next\s+page)$",
    re.IGNORECASE,
)

_NEGATIVE_BUTTON_PATTERNS = re.compile(
    r"\b(cancel|back|previous|prev|delete|remove|close|dismiss|sign\s+in|login|log\s+in|search|filter|menu|share|attach|upload\s+file)\b",
    re.IGNORECASE,
)


def _score_button_candidate(candidate: dict[str, Any]) -> tuple[float, str]:
    """Score a candidate button deterministically using heuristic rules.

    Args:
        candidate: Candidate button dictionary.

    Returns:
        tuple[float, str]: (score, action_type)
    """
    text = (candidate.get("text") or "").strip()
    btn_type = (candidate.get("button_type") or "").lower()
    el_id = (candidate.get("element_id") or "").lower()
    el_class = (candidate.get("element_class") or "").lower()

    # Immediate negative exclusion
    if _NEGATIVE_BUTTON_PATTERNS.search(text):
        return -100.0, "ignore"

    score = 0.0
    action_type = "submit"

    # Exact submit regex
    if _EXACT_SUBMIT_PATTERNS.match(text):
        score += 100.0
        action_type = "submit"
    # Contains submit regex
    elif _CONTAINS_SUBMIT_PATTERNS.search(text):
        score += 80.0
        action_type = "submit"
    # Multi-step Next / Continue
    elif _NEXT_STEP_PATTERNS.match(text) or _NEXT_STEP_PATTERNS.search(text):
        score += 65.0
        action_type = "continue"

    # Attribute bonuses
    if btn_type == "submit":
        score += 25.0

    if "submit" in el_id or "apply" in el_id:
        score += 15.0

    if any(k in el_class for k in ("submit", "apply", "btn-primary", "button-primary", "primary")):
        score += 10.0

    return score, action_type


async def _is_frame_detached(frame: Any) -> bool:
    """Check whether a frame is detached asynchronously or synchronously."""
    if hasattr(frame, "is_detached"):
        try:
            res = frame.is_detached() if callable(frame.is_detached) else frame.is_detached
            if hasattr(res, "__await__") or asyncio.iscoroutine(res):
                return bool(await res)
            return bool(res)
        except Exception:
            return True
    return False


# ---------------------------------------------------------------------------
# Core Extraction & Identification Functions
# ---------------------------------------------------------------------------


async def extract_form_schema(page: Any) -> list[FormFieldSchema]:
    """Inspect and extract visible form fields across the main document and all frames/iframes.

    Args:
        page: Playwright Page or Frame object.

    Returns:
        list[FormFieldSchema]: Extracted form field definitions.
    """
    fields: list[FormFieldSchema] = []

    # Get list of frames if available, otherwise inspect page directly
    frames: Sequence[Any]
    if hasattr(page, "frames") and page.frames:
        frames = page.frames
    else:
        frames = [page]

    for frame_idx, frame in enumerate(frames):
        # Check if frame is detached
        if await _is_frame_detached(frame):
            continue

        try:
            raw_fields = await frame.evaluate(_DOM_EXTRACTOR_JS)
            if not isinstance(raw_fields, list):
                continue

            for item in raw_fields:
                if not isinstance(item, dict):
                    continue
                item["frame_index"] = frame_idx
                field_obj = FormFieldSchema(**item)
                fields.append(field_obj)
        except Exception as exc:
            logger.debug("Failed to extract DOM form fields from frame %d: %s", frame_idx, exc)

    logger.info("Extracted %d form fields across %d frames", len(fields), len(frames))
    return fields


async def identify_submit_button(
    page: Any,
    llm_gateway: Optional[Any] = None,
) -> Optional[SubmitButtonInfo]:
    """Inspect button-like elements across all frames and identify the primary submit/continue button.

    Uses deterministic heuristic matching first. If ambiguous, invokes optional LLM gateway
    to pick the best submit button.

    Args:
        page: Playwright Page or Frame object.
        llm_gateway: Optional ResilientLLMGateway instance for disambiguation.

    Returns:
        Optional[SubmitButtonInfo]: Detected submit button info, or None if not found.
    """
    frames: Sequence[Any]
    if hasattr(page, "frames") and page.frames:
        frames = page.frames
    else:
        frames = [page]

    all_candidates: list[dict[str, Any]] = []

    for frame_idx, frame in enumerate(frames):
        if await _is_frame_detached(frame):
            continue

        try:
            frame_candidates = await frame.evaluate(_BUTTON_EXTRACTOR_JS)
            if not isinstance(frame_candidates, list):
                continue

            for cand in frame_candidates:
                if not isinstance(cand, dict):
                    continue
                cand["frame_index"] = frame_idx
                all_candidates.append(cand)
        except Exception as exc:
            logger.debug("Failed to extract candidate buttons from frame %d: %s", frame_idx, exc)

    if not all_candidates:
        logger.debug("No clickable button candidates found on page.")
        return None

    # Score all candidates deterministically
    scored_candidates: list[tuple[float, str, dict[str, Any]]] = []
    for cand in all_candidates:
        score, action_type = _score_button_candidate(cand)
        if score > 0:
            scored_candidates.append((score, action_type, cand))

    if not scored_candidates:
        logger.debug("No candidate buttons passed positive heuristic threshold.")
        return None

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    top_score, top_action, top_candidate = scored_candidates[0]

    # Check if there is ambiguity requiring LLM gateway
    # Ambiguity criteria: 2+ candidates with close scores (difference <= 15) and top score < 95
    is_ambiguous = (
        len(scored_candidates) > 1
        and (top_score - scored_candidates[1][0]) <= 15
        and top_score < 95.0
    )

    if is_ambiguous and llm_gateway is not None:
        try:
            logger.info("Ambiguous submit button candidates detected; invoking LLM gateway.")
            chosen_index = await _disambiguate_submit_button_with_llm(
                [c[2] for c in scored_candidates[:5]],
                llm_gateway,
            )
            if chosen_index is not None and 0 <= chosen_index < len(scored_candidates):
                top_score, top_action, top_candidate = scored_candidates[chosen_index]
                logger.info("LLM resolved submit button: '%s'", top_candidate.get("text"))
        except Exception as exc:
            logger.warning("LLM submit button disambiguation failed: %s; falling back to heuristic top match.", exc)

    # Normalize confidence to 0.0 - 1.0
    confidence = min(1.0, max(0.1, top_score / 125.0))

    return SubmitButtonInfo(
        frame_index=top_candidate["frame_index"],
        selector=top_candidate["selector"],
        text=top_candidate.get("text", ""),
        button_type=top_candidate.get("button_type", "submit"),
        confidence=round(confidence, 2),
        action_type=top_action,
        element_id=top_candidate.get("element_id"),
        element_class=top_candidate.get("element_class"),
    )


async def _disambiguate_submit_button_with_llm(
    candidates: list[dict[str, Any]],
    llm_gateway: Any,
) -> Optional[int]:
    """Use LLM gateway to select the most appropriate submit/continue button from candidates.

    Args:
        candidates: List of candidate button dicts.
        llm_gateway: LLM gateway instance.

    Returns:
        Optional[int]: 0-based index of chosen candidate, or None.
    """
    options_str = "\n".join(
        f"[{i}] Text: '{c.get('text', '')}', Type: '{c.get('button_type', '')}', Selector: '{c.get('selector', '')}', Class: '{c.get('element_class', '')}'"
        for i, c in enumerate(candidates)
    )
    prompt = (
        "Identify the primary form submit or next/continue button for a job application form.\n\n"
        f"Candidate buttons:\n{options_str}\n\n"
        "Return ONLY the single integer index (e.g. 0, 1, 2) of the winning button."
    )

    if hasattr(llm_gateway, "ask_question"):
        response = await llm_gateway.ask_question(prompt)
    elif hasattr(llm_gateway, "complete"):
        response = await llm_gateway.complete(prompt)
    elif callable(llm_gateway):
        response = await llm_gateway(prompt)
    else:
        return None

    if not response:
        return None

    match = re.search(r"\b(\d+)\b", str(response))
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(candidates):
            return idx

    return None
