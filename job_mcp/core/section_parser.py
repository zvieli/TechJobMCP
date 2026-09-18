"""Universal bilingual (English & Hebrew) Section Parser with boilerplate isolation."""

from dataclasses import dataclass
import html
import re
from typing import Optional

from job_mcp.core.api_client import NON_TECH_ROLE_TERMS, _extract_text_tech_keywords


@dataclass
class JobSections:
    """Discrete parsed sections of a job posting."""

    requirements: str = ""
    responsibilities: str = ""
    company_overview: str = ""
    benefits: str = ""
    raw_other: str = ""


# ---------------------------------------------------------------------------
# Bilingual Section Header Patterns
# Order of evaluation: responsibilities, requirements, benefits, company_overview
# (Responsibilities evaluated before company overview so 'about the role'
# is never mistaken for company 'about' overview).
# ---------------------------------------------------------------------------

_RESPONSIBILITIES_PATTERNS: tuple[str, ...] = (
    # English
    r"responsibilities\b",
    r"key\s+responsibilities\b",
    r"core\s+responsibilities\b",
    r"what\s+you(?:'ll|’ll|\s+will)\s+do\b",
    r"what\s+you\s+do\b",
    r"about\s+the\s+(?:role|job|position)\b",
    r"the\s+role\b",
    r"role\s+(?:overview|description)\b",
    r"your\s+impact\b",
    r"duties\b",
    r"scope\s+of\s+work\b",
    # Hebrew
    r"תחומי\s+אחריות",
    r"תחומי\s+האחריות",
    r"תיאור\s+התפקיד",
    r"תיאור\s+תפקיד",
    r"במסגרת\s+התפקיד",
    r"מהות\s+התפקיד",
    r"הגדרת\s+התפקיד",
)

_REQUIREMENTS_PATTERNS: tuple[str, ...] = (
    # English
    r"(?:job\s+|key\s+|core\s+)?requirements\b",
    r"basic\s+qualifications\b",
    r"preferred\s+qualifications\b",
    r"minimum\s+qualifications\b",
    r"qualifications\b",
    r"what\s+you\s+bring\b",
    r"what\s+you(?:'ll|’ll|\s+will)?\s*bring\b",
    r"what\s+you(?:'ll|’ll|\s+will)?\s*need\b",
    r"who\s+you\s+are\b",
    r"skills\s*(?:&|and)\s*experience\b",
    r"skills\s*(?:&|and)\s*qualifications\b",
    r"(?:key\s+|technical\s+)?skills\b",
    r"must\s+haves?\b",
    r"must-haves?\b",
    r"what\s+we(?:'re|’re|\s+are)?\s*(?:looking\s+for|look\s+for)\b",
    r"candidate\s+profile\b",
    # Hebrew
    r"דרישות\s+התפקיד",
    r"דרישות\s+סף",
    r"דרישות\s+חובה",
    r"דרישות\s+יתרון",
    r"דרישות",
    r"כישורים",
    r"מה\s+אנחנו\s+מחפשים",
    r"מה\s+תביא(?:י)?\s+איתך",
    r"פרופיל\s+המועמד",
    r"פרופיל\s+מועמד",
)

_BENEFITS_PATTERNS: tuple[str, ...] = (
    # English
    r"what\s+we\s+offer\b",
    r"perks\s*(?:&|and)\s*benefits\b",
    r"benefits\s*(?:&|and)\s*perks\b",
    r"compensation\s*(?:&|and)\s*benefits\b",
    r"benefits\b",
    r"perks\b",
    r"why\s+join\s+us\b",
    r"why\s+work\s+(?:with|for|at)\s+us\b",
    # Hebrew
    r"מה\s+אנחנו\s+מציעים",
    r"תנאים\s+והטבות",
    r"הטבות\s+ותנאים",
    r"תנאים",
    r"הטבות",
    r"למה\s+לעבוד\s+אצלנו",
    r"למה\s+להצטרף\s+אלינו",
)

_COMPANY_OVERVIEW_PATTERNS: tuple[str, ...] = (
    # English
    r"about\s+the\s+company\b",
    r"about\s+us\b",
    r"who\s+we\s+are\b",
    r"company\s+overview\b",
    r"company\s+description\b",
    r"our\s+mission\b",
    r"life\s+at\s+[\w\s.-]+",
    r"working\s+at\s+[\w\s.-]+",
    r"join\s+our\s+team\b",
    r"about\s+(?!the\s+(?:role|job|position)\b)[\w\s&.'-]{1,35}",
    # Hebrew
    r"אודות\s+החברה",
    r"אודות",
    r"על\s+החברה",
    r"מי\s+אנחנו",
    r"קצת\s+עלינו",
    r"על\s+הארגון",
)

# Ordered list of (section_key, list_of_patterns)
_SECTION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("responsibilities", _RESPONSIBILITIES_PATTERNS),
    ("requirements", _REQUIREMENTS_PATTERNS),
    ("benefits", _BENEFITS_PATTERNS),
    ("company_overview", _COMPANY_OVERVIEW_PATTERNS),
)

# Precompiled regex patterns for performance
_COMPILED_SECTION_PATTERNS: list[tuple[str, list[re.Pattern[str], re.Pattern[str]]]] = [
    (
        sec_name,
        [
            # pattern_with_remainder: Header: Remainder content
            re.compile(rf"^(?:{pat})\s*[:\-–—]\s*(.*)$", re.IGNORECASE),
            # pattern_standalone: Header alone on line (optional separator)
            re.compile(rf"^(?:{pat})\s*[:\-–—\.]?$", re.IGNORECASE),
        ],
    )
    for sec_name, patterns in _SECTION_SPECS
    for pat in patterns
]


def _match_section_header(line: str) -> tuple[Optional[str], str]:
    """Check if a line represents a section header.

    Returns:
        tuple[Optional[str], str]: (section_name, remainder_content) if matched,
                                   or (None, "") if not a header.
    """
    clean = line.strip()
    if not clean:
        return None, ""

    # Strip markdown symbols at line beginning: #, *, _, -, •, >
    clean_no_md = re.sub(r"^[#*_\-•>\s]+", "", clean)
    # Strip leading numbering like '1. ', '2) '
    clean_no_md = re.sub(r"^\d+[\.\)]\s*", "", clean_no_md).strip()
    # Strip trailing markdown formatting
    clean_no_md = re.sub(r"[*_]+$", "", clean_no_md).strip()

    if not clean_no_md:
        return None, ""

    for sec_name, (pat_remainder, pat_standalone) in _COMPILED_SECTION_PATTERNS:
        m_rem = pat_remainder.match(clean_no_md)
        if m_rem:
            return sec_name, m_rem.group(1).strip()

        m_stand = pat_standalone.match(clean_no_md)
        if m_stand:
            return sec_name, ""

    return None, ""


def _is_header_fragment(text: str) -> bool:
    """Check if a text fragment (e.g. inside <strong> or <h*>) is a section header."""
    sec, _ = _match_section_header(text)
    return sec is not None


def _clean_html_and_extract_lines(text_or_html: str) -> list[str]:
    """Clean HTML or markdown text into discrete lines for section parsing."""
    if not text_or_html or not text_or_html.strip():
        return []

    text = text_or_html

    if "<" in text and ">" in text:
        # Strip script and style blocks entirely
        text = re.sub(r"(?is)<script\b[^>]*>.*?</script>", "", text)
        text = re.sub(r"(?is)<style\b[^>]*>.*?</style>", "", text)

        # Headings h1-h6: place content on isolated lines
        text = re.sub(r"(?i)<h[1-6]\b[^>]*>(.*?)</h[1-6]>", r"\n\1\n", text)

        # Strong and bold tags: if the inner text matches a section header, isolate it
        def _strong_replacer(match: re.Match[str]) -> str:
            inner = match.group(1)
            if _is_header_fragment(inner):
                return f"\n{inner}\n"
            return f" {inner} "

        text = re.sub(r"(?i)<(?:strong|b)\b[^>]*>(.*?)</(?:strong|b)>", _strong_replacer, text)

        # Replace break and list elements
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"(?i)<hr\s*/?>", "\n", text)
        text = re.sub(r"(?i)<li\b[^>]*>", "\n• ", text)

        # Replace block container tags with newlines
        text = re.sub(r"(?i)</?(?:p|div|tr|ul|ol|table|section|article|header|footer)\b[^>]*>", "\n", text)

        # Strip any remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

    # Unescape HTML entities
    text = html.unescape(text)

    # Split into clean lines
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            lines.append(line)

    return lines


def parse_job_sections(text_or_html: str) -> JobSections:
    """Parse raw job description text or HTML into structured JobSections.

    Bilingual English and Hebrew header recognition partitions content into:
    - requirements
    - responsibilities
    - company_overview
    - benefits
    - raw_other (initial headers/unmatched intro)

    Args:
        text_or_html: Raw job description text or HTML string.

    Returns:
        JobSections: Dataclass containing parsed sections.
    """
    lines = _clean_html_and_extract_lines(text_or_html)
    if not lines:
        return JobSections()

    section_content: dict[str, list[str]] = {
        "raw_other": [],
        "company_overview": [],
        "responsibilities": [],
        "requirements": [],
        "benefits": [],
    }

    active_section = "raw_other"

    for line in lines:
        matched_sec, remainder = _match_section_header(line)
        if matched_sec:
            active_section = matched_sec
            if remainder:
                section_content[active_section].append(remainder)
        else:
            section_content[active_section].append(line)

    return JobSections(
        requirements="\n".join(section_content["requirements"]).strip(),
        responsibilities="\n".join(section_content["responsibilities"]).strip(),
        company_overview="\n".join(section_content["company_overview"]).strip(),
        benefits="\n".join(section_content["benefits"]).strip(),
        raw_other="\n".join(section_content["raw_other"]).strip(),
    )


def is_non_tech_role(title: str, department: Optional[str] = None) -> bool:
    """Determine if a job title or department matches known non-technical role terms.

    Args:
        title: Job title string.
        department: Optional department string.

    Returns:
        bool: True if role or department is classified as non-technical.
    """
    check_str = f"{title or ''} {department or ''}".lower()
    return any(
        re.search(r"\b" + re.escape(term) + r"\b", check_str)
        for term in NON_TECH_ROLE_TERMS
    )


def extract_clean_job_tech_stack(
    title: str,
    sections: JobSections,
    department: Optional[str] = None,
    fallback_text: Optional[str] = None,
) -> list[str]:
    """Extract tech stack keywords strictly from requirements and responsibilities.

    Crucial Rule:
        `sections.company_overview` and `sections.benefits` are completely EXCLUDED
        to isolate company boilerplate across all sources.

    Fallback Rule:
        If `sections.requirements` and `sections.responsibilities` are both empty (unstructured text):
        - If `title` or `department` matches `NON_TECH_ROLE_TERMS`, returns []
          (guarantees non-tech roles never receive false-positive tech stacks).
        - Otherwise, extracts keywords from title, department, and fallback text.

    Args:
        title: Job title string.
        sections: Parsed JobSections instance.
        department: Optional department string.
        fallback_text: Optional fallback text used when sections are unstructured.

    Returns:
        list[str]: Extracted tech stack keywords.
    """
    req = (sections.requirements or "").strip()
    resp = (sections.responsibilities or "").strip()

    if not req and not resp:
        if is_non_tech_role(title, department):
            return []
        fallback = fallback_text if fallback_text is not None else sections.raw_other
        target = f"{title or ''} {department or ''} {fallback or ''}".strip()
        return _extract_text_tech_keywords(target)

    # Structured sections available: strictly exclude company_overview and benefits
    target_text = f"{title or ''} {department or ''} {req} {resp}".strip()
    return _extract_text_tech_keywords(target_text)
