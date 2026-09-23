"""Bootstrap training and evaluation datasets for TechJobMCP v2 dual-cognition Laya models.

Generates labeled datasets across 5 domains:
1. match_scoring: (job_desc, candidate_cv) -> skill_match (0-4), seniority_fit (0-4), recruiter_fit (bool)
2. dedup_verification: (job_a, job_b) -> is_duplicate (bool)
3. role_classification: (job_title, job_desc) -> category (Core Engineering / Engineering-Adjacent / Technical Hybrid / Non-Technical / Administrative)
4. section_parsing: (paragraph) -> section_type (Requirements / Responsibilities / Company overview / Benefits / Application instructions / Boilerplate)
5. field_mapping: (field_label, placeholder, type) -> field_intent (First name / Last name / Email / Phone / Resume / Cover letter / Custom / etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from job_mcp.core.llm.gateway import ResilientLLMGateway

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_training_data")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR = DATA_DIR / "training"
HOLDOUT_DIR = DATA_DIR / "holdout"


async def generate_batch_with_llm(
    gateway: ResilientLLMGateway,
    task_prompt: str,
    system_prompt: str,
    expected_count: int,
) -> List[Dict[str, Any]]:
    """Prompt the LLM gateway to generate structured synthetic examples in JSON."""
    full_prompt = (
        f"{task_prompt}\n\n"
        f"Generate exactly {expected_count} diverse, realistic examples. "
        "Include realistic tech stack combinations, edge cases, both English and Hebrew listings.\n"
        "Return ONLY a valid JSON array of objects, with no markdown code fences and no conversational filler."
    )

    # Use gateway internal call directly to avoid screening question cache/formatting
    providers = gateway._get_provider_chain()
    if not providers:
        raise RuntimeError("No LLM providers available in gateway chain.")

    last_err = None
    for name, fn in providers:
        try:
            logger.info("Requesting batch generation via provider: %s", name)
            raw = await gateway._execute_with_retry(name, lambda f=fn: f(full_prompt, system_prompt))
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
            logger.warning("Provider %s did not return a JSON array", name)
        except Exception as e:
            logger.warning("Provider %s failed during batch gen: %s", name, e)
            last_err = e

    raise RuntimeError(f"All providers failed to generate batch: {last_err}")


async def generate_match_scoring_samples(
    gateway: ResilientLLMGateway, count: int
) -> List[Dict[str, Any]]:
    """Generate (job_description, candidate_cv) pairs with 3-question ensemble labels.

    Enforces a realistic distribution:
    - ~35% strong matches (skill_match 3-4, seniority 2-3, recruiter_fit true)
    - ~35% hard negatives / near misses (e.g. Java dev applying to Python role; Junior applying to Staff;
      Senior applying to entry level; frontend dev applying to DevOps)
    - ~30% partial matches (skill_match 1-2, seniority 1-2, recruiter_fit false or borderline)
    """
    prompt = """Generate job-candidate evaluation samples for training an AI recruiter triage model.
CRITICAL: Do NOT generate only perfect matches! Enforce this distribution across the batch:
1. Hard Negatives / Near Misses (~35%):
   - Candidate has the right seniority but entirely wrong core language (e.g. 7-year Java/Spring dev applying for Python/FastAPI ML role).
   - Junior candidate (1-2 years) applying for Senior/Principal role (seniority_fit: 0, recruiter_fit: false).
   - Overqualified / Director-level applying for Junior developer (seniority_fit: 4, recruiter_fit: false).
   - Domain mismatch (e.g. Frontend React developer applying for Linux Kernel / Embedded C role).
2. Partial Matches (~30%):
   - Candidate knows 1-2 secondary tools (e.g. Docker, Git) but lacks primary stack (e.g. no Go experience for Senior Go role).
3. Strong Matches (~35%):
   - Direct stack and seniority alignment.

For each sample, provide:
- "job_title": string
- "job_description": 2-4 sentences describing requirements, tech stack, and seniority
- "candidate_cv": 2-4 sentences describing candidate's stack, experience, projects, and seniority
- "skill_match": integer 0-4 (0: None, 1: Weak, 2: Partial, 3: Strong, 4: Perfect)
- "seniority_fit": integer 0-4 (0: Far too junior, 1: Slightly junior, 2: Good fit, 3: Senior, 4: Overqualified)
- "recruiter_fit": boolean (true if candidate would realistically get phone screen, false otherwise)
- "match_type": "strong_match" | "hard_negative" | "partial_match"
- "reasoning": 1-2 sentences explaining why the recruiter made this decision"""

    system = "You are an expert technical recruiter calibrating an AI job matching dataset. You are rigorous and reject unqualified candidates."
    return await generate_batch_with_llm(gateway, prompt, system, count)


async def generate_dedup_samples(
    gateway: ResilientLLMGateway, count: int
) -> List[Dict[str, Any]]:
    """Generate (job_a, job_b) pairs with duplicate verification labels.

    Enforces:
    - ~40% true duplicates (cross-postings with title variations, different ATS formatting, same company)
    - ~40% hard negatives (same company different seniority, same company different department, same title different company)
    - ~20% obvious negatives
    """
    prompt = """Generate pairs of job postings to train a cross-platform duplicate detection engine.
CRITICAL: Include subtle hard negatives that string matching or naive algorithms fail on!
Required distribution:
1. True Duplicates (~40%):
   - Cross-postings across platforms (e.g. "Senior Python Engineer" on Comeet vs "Sr. Software Engineer - Python" on LinkedIn at the same company).
   - Hebrew vs English titles for the exact same opening (e.g. "מפתח Fullstack" vs "Full Stack Developer" at same company).
   - Same job posted with slightly different location labels (e.g. "Tel Aviv (Hybrid)" vs "Israel - Central").
2. Hard Negatives (~40%):
   - SAME company, SAME tech stack, but DIFFERENT seniority (e.g. "Junior Backend Developer" vs "Tech Lead Backend" at Monday.com).
   - SAME company, DIFFERENT role (e.g. "Product Manager - AI" vs "AI Research Scientist" at same company).
   - SAME exact title, DIFFERENT company (e.g. "Senior DevOps Engineer" at Wix vs "Senior DevOps Engineer" at AppsFlyer).
   - Parent company vs subsidiary listing different requisitions.
3. Obvious Negatives (~20%): Completely different companies and titles.

For each sample, provide:
- "job_a": {"title": string, "company": string, "location": string}
- "job_b": {"title": string, "company": string, "location": string}
- "is_duplicate": boolean (true if these represent the exact same opening, false otherwise)
- "pair_type": "cross_platform_dup" | "same_co_diff_role" | "same_title_diff_co" | "diff_all"
- "reasoning": brief explanation"""

    system = "You are an ATS data engineer specializing in job aggregation deduplication and anti-collision."
    return await generate_batch_with_llm(gateway, prompt, system, count)


async def generate_role_classification_samples(
    gateway: ResilientLLMGateway, count: int
) -> List[Dict[str, Any]]:
    """Generate role title/description samples with 5-class categorization."""
    prompt = """Generate job titles and snippets for role category classification.
Categories MUST be one of:
["Core Engineering", "Engineering-Adjacent", "Technical Hybrid", "Non-Technical", "Administrative"]

CRITICAL: Emphasize boundary and edge cases:
- Technical Hybrid: Solutions Architect, Developer Advocate / DevRel, Technical Product Manager (TPM), Forward Deployed Engineer.
- Engineering-Adjacent: Data Analyst, BI Developer, QA Manual, Technical Support Tier 3, SRE Operations.
- Administrative: Scrum Master, Agile Coach, IT Helpdesk, Project Coordinator.
- Non-Technical: HR Recruiter, Sales Executive, Marketing Manager, Legal Counsel.
- Core Engineering: Backend, Fullstack, Frontend, Embedded, ML Engineer, DevOps Infrastructure.

Include Hebrew titles (e.g. "מנהל מוצר טכנולוגי", "איש סיסטם ותמיכה", "מהנדס אלגוריתמים").

For each sample, provide:
- "title": string
- "snippet": 1-2 sentence description
- "category": one of the 5 categories above
- "reasoning": brief explanation"""

    system = "You are an AI taxonomy specialist classifying job postings into career categories."
    return await generate_batch_with_llm(gateway, prompt, system, count)


async def generate_section_parsing_samples(
    gateway: ResilientLLMGateway, count: int
) -> List[Dict[str, Any]]:
    """Generate job description paragraphs with section type labels."""
    prompt = """Generate individual paragraphs or snippets from diverse job postings (50% English, 50% Hebrew).
Section types MUST be one of:
["Requirements", "Responsibilities", "Company overview", "Benefits", "Application instructions", "Boilerplate"]

CRITICAL: Include unstructured and tricky formats:
- Bullet points WITHOUT clear section headers (e.g. starting directly with "- 3+ years experience...").
- Mixed content (e.g. paragraph describing company mission while casually mentioning required degree).
- Hebrew listings with colloquial phrasing (e.g. "מה אנחנו מציעים?", "מה נדרש ממך?", "קצת עלינו").
- Legal boilerplate and EEO statements.

For each sample, provide:
- "paragraph": text of the snippet
- "section_type": one of the 6 section types above
- "language": "en" | "he"
- "has_explicit_header": boolean"""

    system = "You are an NLP engineer parsing unstructured ATS job postings without relying on regex headers."
    return await generate_batch_with_llm(gateway, prompt, system, count)


async def generate_field_mapping_samples(
    gateway: ResilientLLMGateway, count: int
) -> List[Dict[str, Any]]:
    """Generate job application form fields with target intent labels."""
    prompt = """Generate form field inputs found in various ATS application forms (Comeet, Greenhouse, Lever, Workday).
Field intents MUST be one of:
["First name", "Last name", "Full name", "Email", "Phone", "LinkedIn", "GitHub", "Resume upload", "Cover letter", "Years of experience", "Education", "Work authorization", "Salary expectation", "Custom screening question", "Other"]

CRITICAL: Include non-standard, cryptic, and Hebrew labels:
- Ambiguous labels (e.g. "Tell us about a time...", "Links / Portfolio", "CV / Attachment").
- Hebrew labels (e.g. "טלפון נייד", "שם מלא", "ציפיות שכר חודשיות ברוטו", "האם יש ברשותך אישור עבודה תקף?").
- HTML placeholder cues (e.g. placeholder "https://...", "e.g. 35,000 NIS").
- Screening questions (e.g. "Are you willing to work 3 days from the office in Herzliya?").

For each sample, provide:
- "label": field label text
- "placeholder": optional placeholder text
- "field_type": e.g. "text", "textarea", "select", "file", "radio"
- "field_intent": one of the 15 intents above"""

    system = "You are an automation engineer mapping HTML web form fields to candidate profile attributes."
    return await generate_batch_with_llm(gateway, prompt, system, count)


async def run_task_loop(
    gateway: ResilientLLMGateway,
    name: str,
    gen_fn: Any,
    target_count: int,
    batch_size: int,
    holdout_ratio: float,
) -> None:
    """Run iterative generation loop for a single task until target count is reached."""
    train_file = TRAINING_DIR / f"{name}_train.json"
    holdout_file = HOLDOUT_DIR / f"{name}_holdout.json"

    existing_train: List[Dict[str, Any]] = (
        json.loads(train_file.read_text()) if train_file.exists() else []
    )
    existing_holdout: List[Dict[str, Any]] = (
        json.loads(holdout_file.read_text()) if holdout_file.exists() else []
    )

    total_existing = len(existing_train) + len(existing_holdout)
    if total_existing >= target_count:
        logger.info(
            "Task %s already has %d samples (target: %d). Skipping.",
            name,
            total_existing,
            target_count,
        )
        return

    logger.info(
        "Starting generation for %s: %d existing, %d target (need %d more)...",
        name,
        total_existing,
        target_count,
        target_count - total_existing,
    )

    while (len(existing_train) + len(existing_holdout)) < target_count:
        needed = target_count - (len(existing_train) + len(existing_holdout))
        current_batch = min(batch_size, needed)

        try:
            samples = await gen_fn(gateway, current_batch)
            if not samples:
                logger.warning("Empty batch received for %s. Retrying in 2s...", name)
                await asyncio.sleep(2.0)
                continue

            random.shuffle(samples)
            split_idx = int(len(samples) * (1.0 - holdout_ratio))
            train_part = samples[:split_idx]
            holdout_part = samples[split_idx:]

            existing_train.extend(train_part)
            existing_holdout.extend(holdout_part)

            train_file.write_text(json.dumps(existing_train, indent=2, ensure_ascii=False))
            holdout_file.write_text(json.dumps(existing_holdout, indent=2, ensure_ascii=False))

            total_now = len(existing_train) + len(existing_holdout)
            logger.info(
                "[%s Progress] %d / %d samples generated (Train: %d, Holdout: %d)",
                name,
                total_now,
                target_count,
                len(existing_train),
                len(existing_holdout),
            )

            # Polite delay between batches to respect rate limits
            await asyncio.sleep(1.5)

        except Exception as e:
            logger.error("Error during batch generation for %s: %s. Pausing 5s...", name, e)
            await asyncio.sleep(5.0)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data for TechJobMCP v2 Laya models")
    parser.add_argument("--target-per-task", type=int, default=400, help="Target total samples per task (default: 400 -> 2,000 total)")
    parser.add_argument("--batch-size", type=int, default=15, help="Number of items to request per LLM batch call")
    parser.add_argument("--holdout-ratio", type=float, default=0.2, help="Ratio of samples to reserve for holdout set")
    parser.add_argument("--task", type=str, default="all", help="Specific task to run (match_scoring, dedup_verification, role_classification, section_parsing, field_mapping, or all)")
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    gateway = ResilientLLMGateway()

    task_map = {
        "match_scoring": generate_match_scoring_samples,
        "dedup_verification": generate_dedup_samples,
        "role_classification": generate_role_classification_samples,
        "section_parsing": generate_section_parsing_samples,
        "field_mapping": generate_field_mapping_samples,
    }

    selected_tasks = (
        task_map.items() if args.task == "all" else [(args.task, task_map[args.task])]
    )

    for name, gen_fn in selected_tasks:
        await run_task_loop(
            gateway=gateway,
            name=name,
            gen_fn=gen_fn,
            target_count=args.target_per_task,
            batch_size=args.batch_size,
            holdout_ratio=args.holdout_ratio,
        )

    logger.info("All requested generation tasks completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
