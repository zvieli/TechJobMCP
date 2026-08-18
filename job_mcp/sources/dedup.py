"""Deduplication and entity merging engine for multi-source job aggregation."""

import re
from typing import Optional
from job_mcp.models.schemas import Job

ATS_DOMAINS = (
    "comeet.com",
    "comeet.me",
    "greenhouse.io",
    "lever.co",
    "workday.com",
    "myworkdayjobs.com",
    "smartrecruiters.com",
    "ashbyhq.com",
    "bamboohr.com",
    "jobvite.com",
    "workable.com",
    "recruitee.com",
    "rippling.com",
    "personio.com",
    "pinpointhq.com",
    "taleo.net",
    "icims.com",
    "successfactors.com",
    "breezy.hr",
    "applicantlist.com",
)

AGGREGATOR_DOMAINS = (
    "alljobs.co.il",
    "jobmaster.co.il",
    "drushim.co.il",
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "drushim.gov.il",
)

PRIMARY_DOMAINS = (
    "hiremetech.com",
    "hireme.tech",
)

CORPORATE_SUFFIXES = {
    "ltd",
    "inc",
    "llc",
    "corp",
    "corporation",
    "co",
    "company",
    "limited",
    "incorporated",
    "gmbh",
    "bv",
    "בעמ",
    "בע מ",
    "technologies",
    "tech",
    "group",
    "holdings",
    "טכנולוגיות",
    "טכנולוגיה",
    "קבוצת",
}


def normalize_title(title: str) -> str:
    """Normalize a job title for deduplication.

    - Lowercases text
    - Strips gender annotations: (m/f/d), (ז/נ), (m/f), מפתח/ת, etc.
    - Strips work-mode noise: (hybrid), (remote), (onsite), היברידי, etc.
    - Strips common filler keywords while preserving primary tech/role words.
    - Strips punctuation and collapses whitespace.
    """
    if not title or not isinstance(title, str):
        return ""

    t = title.lower()

    # Protect tech keywords with special symbols using non-word/boundary lookarounds
    t = re.sub(r'\bc\+\+(?=[^a-zA-Z0-9]|$)', '__cpp_token__', t)
    t = re.sub(r'\bc#(?=[^a-zA-Z0-9]|$)', '__csharp_token__', t)
    t = re.sub(r'\bf#(?=[^a-zA-Z0-9]|$)', '__fsharp_token__', t)
    t = re.sub(r'(?:\b|\.)net(?=[^a-zA-Z0-9]|$)', '__dotnet_token__', t)
    t = re.sub(r'\bnode\.js(?=[^a-zA-Z0-9]|$)', '__nodejs_token__', t)
    t = re.sub(r'\breact\.js(?=[^a-zA-Z0-9]|$)', 'react', t)
    t = re.sub(r'\bvue\.js(?=[^a-zA-Z0-9]|$)', 'vue', t)
    t = re.sub(r'\bnext\.js(?=[^a-zA-Z0-9]|$)', 'nextjs', t)

    # Strip Hebrew slash gender forms: מפתח/ת -> מפתח, מהנדס/ת -> מהנדס, איש/אשת -> איש
    t = re.sub(r'([\u0590-\u05FF]+)/[תתה]\b', r'\1', t)
    t = re.sub(r'\b(איש)/אשת\b', r'\1', t)
    t = re.sub(r'\b(ראש)/ת\b', r'\1', t)

    # Strip gender annotations in parentheses/brackets or standalone
    t = re.sub(r'[\(\[\{]\s*(m/f/d|m/w/d|f/m/d|m/f|f/m|d/m/f|ז/נ|נ/ז|זכר/נקבה|זכר|נקבה)\s*[\)\]\}]', ' ', t)
    t = re.sub(r'\b(m/f/d|m/w/d|f/m/d|m/f|f/m|ז/נ|נ/ז)\b', ' ', t)

    # Strip work mode indicators
    t = re.sub(r'[\(\[\{]\s*(hybrid|remote|onsite|on-site|היברידית?|מרחוק|מהבית|באתר|מהמשרד)\s*[\)\]\}]', ' ', t)
    t = re.sub(r'\b(hybrid|remote|onsite|on-site|היברידית?|מרחוק|מהבית|באתר|מהמשרד)\b', ' ', t)

    # Strip filler keywords
    filler_patterns = [
        r'\[urgent\]',
        r'\burgent\b',
        r'\bimmediate\s+opening\b',
        r'\bimmediate\b',
        r'\bwe\s+are\s+hiring\b',
        r'\bwe\'re\s+hiring\b',
        r'\bhiring\b',
        r'\bwanted\b',
        r'\blooking\s+for\b',
        r'\bדרוש\b',
        r'\bדרושה\b',
        r'\bדרושים\b',
        r'\bדרושות\b',
        r'\bמשרה\s+מלאה\b',
        r'\bמשרה\s+חלקית\b',
        r'\bfull\s+time\b',
        r'\bpart\s+time\b',
        r'\bfull-time\b',
        r'\bpart-time\b',
    ]
    for pattern in filler_patterns:
        t = re.sub(pattern, ' ', t)

    # Strip punctuation (keep word characters, Hebrew letters, whitespace, and protected tokens)
    t = re.sub(r'[^\w\s\u0590-\u05FF]', ' ', t)

    # Restore tech tokens
    t = t.replace('__cpp_token__', 'c++')
    t = t.replace('__csharp_token__', 'c#')
    t = t.replace('__fsharp_token__', 'f#')
    t = t.replace('__dotnet_token__', '.net')
    t = t.replace('__nodejs_token__', 'nodejs')

    return " ".join(t.split())


def normalize_company(company: str) -> str:
    """Normalize a company name for deduplication.

    - Lowercases text
    - Removes web domain extensions (.com, .io, .ai, etc.)
    - Removes corporate suffixes: ltd, inc, llc, corp, בע"מ, בעמ, technologies, tech, group, etc.
    - Strips punctuation and collapses whitespace.
    """
    if not company or not isinstance(company, str):
        return ""

    c = company.lower()

    # Strip domain extensions
    c = re.sub(r'\.(com|co\.il|io|ai|org|net|co|me|dev)\b', ' ', c)

    # Remove Hebrew corporate suffixes
    c = re.sub(r'בע["״\']מ', ' ', c)
    c = re.sub(r'בע\s+מ', ' ', c)
    c = re.sub(r'\bבעמ\b', ' ', c)

    # Strip punctuation
    c = re.sub(r'[^\w\s\u0590-\u05FF]', ' ', c)

    tokens = c.split()
    if not tokens:
        return ""

    # Filter corporate suffix tokens
    filtered = [t for t in tokens if t not in CORPORATE_SUFFIXES]

    if filtered:
        return " ".join(filtered)
    # If all tokens were suffixes (e.g. "Tech Corp" or "Group Ltd"), preserve the first token
    return tokens[0]


def compute_dedup_key(title: str, company: str) -> str:
    """Compute deduplication key for a job listing based on normalized title and company."""
    norm_title = normalize_title(title)
    norm_company = normalize_company(company)
    return f"{norm_title}@{norm_company}"


def _get_url_priority(url: Optional[str]) -> int:
    """Compute priority score for a URL:

    - Direct ATS systems (Comeet, Greenhouse, Lever, Workday, etc.): 100
    - Company career pages: 80
    - HireMeTech: 50
    - Generic external links: 30
    - Job aggregators / boards (AllJobs, JobMaster, Drushim, LinkedIn, etc.): 20
    - None / empty: 0
    """
    if not url or not isinstance(url, str):
        return 0

    url_lower = url.lower()

    for ats in ATS_DOMAINS:
        if ats in url_lower:
            return 100

    if any(kw in url_lower for kw in ("careers.", "jobs.", "/careers/", "/jobs/apply")) and not any(
        agg in url_lower for agg in AGGREGATOR_DOMAINS
    ):
        return 80

    for primary in PRIMARY_DOMAINS:
        if primary in url_lower:
            return 50

    for agg in AGGREGATOR_DOMAINS:
        if agg in url_lower:
            return 20

    return 30


def merge_job_entities(primary: Job, secondary: Job) -> Job:
    """Merge two duplicate Job entities into a single unified Job model.

    - Combines sources list (preserving order and uniqueness).
    - Prefers ATS/direct apply links for apply_url and url.
    - Combines and deduplicates tech_stack.
    - Uses the longer / more comprehensive description.
    - Retains best available salary_range, posted_date, location, work_mode, department, etc.
    """
    # 1. Sources union
    sources_combined: list[str] = []
    for s in (primary.sources or []) + [primary.source] + (secondary.sources or []) + [secondary.source]:
        if s and s not in sources_combined:
            sources_combined.append(s)

    # 2. Tech stack union
    tech_seen = set()
    merged_tech: list[str] = []
    for tech in (primary.tech_stack or []) + (secondary.tech_stack or []):
        if tech and tech.lower() not in tech_seen:
            tech_seen.add(tech.lower())
            merged_tech.append(tech)

    # 3. Description: prefer longer / more comprehensive
    desc_primary = (primary.description or "").strip()
    desc_secondary = (secondary.description or "").strip()
    merged_desc = desc_primary if len(desc_primary) >= len(desc_secondary) else desc_secondary

    # 4. URLs: evaluate priority for apply_url and url
    # Candidate apply URLs
    apply_candidates = [u for u in [primary.apply_url, secondary.apply_url] if u]
    if apply_candidates:
        best_apply_url = max(apply_candidates, key=_get_url_priority)
    else:
        # Fallback: check if url is ATS
        all_urls = [u for u in [primary.url, secondary.url] if u]
        ats_urls = [u for u in all_urls if _get_url_priority(u) >= 80]
        best_apply_url = ats_urls[0] if ats_urls else None

    # Best general url
    url_candidates = [u for u in [primary.url, secondary.url] if u]
    if url_candidates:
        best_url = max(url_candidates, key=_get_url_priority)
    else:
        best_url = None

    # 5. Metadata fields
    salary_range = primary.salary_range or secondary.salary_range
    posted_date = primary.posted_date or secondary.posted_date
    location = primary.location if (primary.location and primary.location.strip()) else secondary.location
    work_mode = primary.work_mode or secondary.work_mode
    department = primary.department or secondary.department
    is_bookmarked = primary.is_bookmarked or secondary.is_bookmarked

    # Match score
    if primary.match_score is not None and secondary.match_score is not None:
        match_score = max(primary.match_score, secondary.match_score)
    elif primary.match_score is not None:
        match_score = primary.match_score
    else:
        match_score = secondary.match_score

    # Explainability & Enrichment fields
    matched_skills = list(dict.fromkeys(primary.matched_skills + secondary.matched_skills))
    missing_skills = list(dict.fromkeys(primary.missing_skills + secondary.missing_skills))
    match_reasons = list(dict.fromkeys(primary.match_reasons + secondary.match_reasons))
    description_summary = primary.description_summary or secondary.description_summary
    seniority_level = primary.seniority_level or secondary.seniority_level

    return Job(
        job_id=primary.job_id,
        title=primary.title if primary.title.strip() else secondary.title,
        company=primary.company if primary.company.strip() else secondary.company,
        location=location,
        work_mode=work_mode,
        tech_stack=merged_tech,
        description=merged_desc,
        salary_range=salary_range,
        posted_date=posted_date,
        url=best_url,
        is_bookmarked=is_bookmarked,
        match_score=match_score,
        source=primary.source,
        sources=sources_combined,
        apply_url=best_apply_url,
        department=department,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        match_reasons=match_reasons,
        description_summary=description_summary,
        seniority_level=seniority_level,
    )


def deduplicate_jobs(jobs: list[Job]) -> list[Job]:
    """Deduplicate a list of jobs across multiple sources by merging duplicate entities."""
    if not jobs:
        return []

    merged_map: dict[str, Job] = {}
    for job in jobs:
        key = compute_dedup_key(job.title, job.company)
        if key in merged_map:
            merged_map[key] = merge_job_entities(merged_map[key], job)
        else:
            merged_map[key] = job

    return list(merged_map.values())
