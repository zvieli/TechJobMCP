#!/usr/bin/env python3
"""
scripts/prepare_laya_dataset.py
--------------------------------
Converts generated synthetic datasets from raw domain JSON into transparent,
human-readable JSONL files for Laya model fine-tuning.

Tasks converted:
1. match_scoring -> 3 items:
   - match_scoring_skill (score: 5 options)
   - match_scoring_seniority (score: 5 options)
   - match_scoring_recruiter_fit (noul: 2 options)
2. dedup_verification -> noul (2 options)
3. role_classification -> choice (5 options)
4. section_parsing -> choice (6 options)
5. field_mapping -> choice (13-15 options)

Outputs:
- data/training/laya_train.jsonl
- data/holdout/laya_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_laya_dataset")

# Fixed option vocabularies for categorical / choice tasks
ROLE_CATEGORIES = [
    "Core Engineering",
    "Engineering-Adjacent",
    "Technical Hybrid",
    "Administrative",
    "Non-Technical",
]

SECTION_TYPES = [
    "Requirements",
    "Responsibilities",
    "Company overview",
    "Benefits",
    "Application instructions",
    "Boilerplate",
]

FIELD_INTENTS = [
    "Full name",
    "First name",
    "Last name",
    "Email",
    "Resume upload",
    "Cover letter",
    "LinkedIn",
    "GitHub",
    "Work authorization",
    "Years of experience",
    "Education",
    "Custom screening question",
    "Other",
]

SKILL_OPTIONS = [
    "None (0-20%)",
    "Weak (20-40%)",
    "Partial (40-60%)",
    "Strong (60-80%)",
    "Perfect (80-100%)",
]

SENIORITY_OPTIONS = [
    "Far too junior",
    "Slightly junior",
    "Good fit",
    "Senior",
    "Overqualified",
]

BINARY_OPTIONS = ["No", "Yes"]
DEDUP_OPTIONS = ["Different jobs", "Duplicate jobs"]


def convert_match_scoring(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand match scoring sample into 3 discrete Laya training items."""
    results = []
    job_title = item.get("job_title", "")
    job_desc = item.get("job_description", "")
    candidate_cv = item.get("candidate_cv", "")

    state_context = f"Job Title: {job_title}\nJob Description:\n{job_desc}\nCandidate CV:\n{candidate_cv}"

    # 1. Skill Match (Score: 0..4)
    skill_val = int(item.get("skill_match", 0))
    skill_val = max(0, min(4, skill_val))
    results.append({
        "task": "match_scoring_skill",
        "qtype": "score",
        "state": state_context,
        "instruction": "Evaluate the technical and professional skill match of the candidate for this role.",
        "options": SKILL_OPTIONS,
        "label": skill_val,
        "metadata": {"job_title": job_title},
    })

    # 2. Seniority Fit (Score: 0..4)
    sen_val = int(item.get("seniority_fit", 2))
    sen_val = max(0, min(4, sen_val))
    results.append({
        "task": "match_scoring_seniority",
        "qtype": "score",
        "state": state_context,
        "instruction": "Assess the seniority alignment of the candidate relative to the requirements.",
        "options": SENIORITY_OPTIONS,
        "label": sen_val,
        "metadata": {"job_title": job_title},
    })

    # 3. Recruiter Fit (Noul: binary 0/1)
    rec_val = 1 if item.get("recruiter_fit", False) else 0
    results.append({
        "task": "match_scoring_recruiter_fit",
        "qtype": "noul",
        "state": state_context,
        "instruction": "Would a human technical recruiter recommend advancing this candidate to an interview?",
        "options": BINARY_OPTIONS,
        "label": rec_val,
        "metadata": {"job_title": job_title},
    })

    return results


def convert_dedup_verification(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert deduplication sample into noul binary hypothesis."""
    job_a = item.get("job_a", {})
    job_b = item.get("job_b", {})
    is_dup = 1 if item.get("is_duplicate", False) else 0

    state_context = (
        f"Job A:\n"
        f"Title: {job_a.get('title', '')}\n"
        f"Company: {job_a.get('company', '')}\n"
        f"Location: {job_a.get('location', '')}\n\n"
        f"Job B:\n"
        f"Title: {job_b.get('title', '')}\n"
        f"Company: {job_b.get('company', '')}\n"
        f"Location: {job_b.get('location', '')}"
    )

    return [{
        "task": "dedup_verification",
        "qtype": "noul",
        "state": state_context,
        "instruction": "Determine whether Job A and Job B describe the exact same underlying job posting.",
        "options": DEDUP_OPTIONS,
        "label": is_dup,
        "metadata": {"reasoning": item.get("reasoning", "")},
    }]


def convert_role_classification(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert role classification sample into choice primitive."""
    title = item.get("title", "")
    snippet = item.get("snippet", "")
    category = item.get("category", "Non-Technical")

    try:
        label = ROLE_CATEGORIES.index(category)
    except ValueError:
        label = ROLE_CATEGORIES.index("Non-Technical")

    state_context = f"Job Title: {title}\nJob Snippet: {snippet}"

    return [{
        "task": "role_classification",
        "qtype": "choice",
        "state": state_context,
        "instruction": "Classify this job title and snippet into the appropriate career role category.",
        "options": ROLE_CATEGORIES,
        "label": label,
        "metadata": {"category": category},
    }]


def convert_section_parsing(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert section parsing sample into choice primitive."""
    paragraph = item.get("paragraph", "")
    section_type = item.get("section_type", "Boilerplate")

    try:
        label = SECTION_TYPES.index(section_type)
    except ValueError:
        label = SECTION_TYPES.index("Boilerplate")

    state_context = f"Text Paragraph:\n{paragraph}"

    return [{
        "task": "section_parsing",
        "qtype": "choice",
        "state": state_context,
        "instruction": "Identify the primary job description section type that this paragraph belongs to.",
        "options": SECTION_TYPES,
        "label": label,
        "metadata": {"section_type": section_type},
    }]


def convert_field_mapping(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert field mapping sample into choice primitive."""
    label_text = item.get("label", "")
    placeholder = item.get("placeholder", "")
    field_type = item.get("field_type", "")
    field_intent = item.get("field_intent", "Other")

    try:
        label_idx = FIELD_INTENTS.index(field_intent)
    except ValueError:
        label_idx = FIELD_INTENTS.index("Other")

    state_context = (
        f"Input Label: {label_text}\n"
        f"Placeholder: {placeholder}\n"
        f"HTML Input Type: {field_type}"
    )

    return [{
        "task": "field_mapping",
        "qtype": "choice",
        "state": state_context,
        "instruction": "Map this job application form input element to its standard semantic field intent.",
        "options": FIELD_INTENTS,
        "label": label_idx,
        "metadata": {"field_intent": field_intent},
    }]


def process_dataset(directory: Path, split_suffix: str) -> List[Dict[str, Any]]:
    """Load all task JSON files for a split and convert to Laya items."""
    converted: List[Dict[str, Any]] = []

    converters = {
        f"match_scoring_{split_suffix}.json": convert_match_scoring,
        f"dedup_verification_{split_suffix}.json": convert_dedup_verification,
        f"role_classification_{split_suffix}.json": convert_role_classification,
        f"section_parsing_{split_suffix}.json": convert_section_parsing,
        f"field_mapping_{split_suffix}.json": convert_field_mapping,
    }

    for fname, converter in converters.items():
        file_path = directory / fname
        if not file_path.exists():
            logger.warning(f"File {file_path} not found. Skipping.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        count_before = len(converted)
        for item in items:
            converted.extend(converter(item))
        count_added = len(converted) - count_before
        logger.info(f"Loaded {len(items)} samples from {fname} -> produced {count_added} Laya items.")

    return converted


def main():
    parser = argparse.ArgumentParser(description="Prepare human-readable Laya training & validation JSONL datasets.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/training"), help="Path to raw training JSON files")
    parser.add_argument("--holdout-dir", type=Path, default=Path("data/holdout"), help="Path to raw holdout JSON files")
    parser.add_argument("--out-train", type=Path, default=Path("data/training/laya_train.jsonl"), help="Output training JSONL")
    parser.add_argument("--out-val", type=Path, default=Path("data/holdout/laya_val.jsonl"), help="Output validation JSONL")
    args = parser.parse_args()

    # Process training split
    logger.info("Processing training split...")
    train_items = process_dataset(args.train_dir, "train")
    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(train_items)} training items to {args.out_train}")

    # Process holdout split
    logger.info("Processing holdout split...")
    val_items = process_dataset(args.holdout_dir, "holdout")
    args.out_val.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_val, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(val_items)} validation items to {args.out_val}")

    # Display task distribution summary
    def print_summary(items: List[Dict[str, Any]], name: str):
        task_counts: Dict[str, int] = {}
        for x in items:
            t = x["task"]
            task_counts[t] = task_counts.get(t, 0) + 1
        print(f"\n--- {name} Dataset Summary ({len(items)} total) ---")
        for t, c in sorted(task_counts.items()):
            print(f"  {t:30s}: {c:4d} items")

    print_summary(train_items, "TRAIN")
    print_summary(val_items, "VALIDATION / HOLDOUT")


if __name__ == "__main__":
    main()
