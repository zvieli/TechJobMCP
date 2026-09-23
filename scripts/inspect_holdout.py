"""Interactive inspection tool for reviewing and verifying generated holdout samples.

Allows the user to inspect generated holdout examples across all domains,
view the ground truth labels, and confirm sample quality.
"""

from __future__ import annotations

import json
from pathlib import Path

HOLDOUT_DIR = Path(__file__).resolve().parent.parent / "data" / "holdout"


def inspect_samples(category: str, limit: int = 5) -> None:
    path = HOLDOUT_DIR / f"{category}_holdout.json"
    if not path.exists():
        print(f"[!] No holdout file found for category: {category}")
        return

    data = json.loads(path.read_text())
    print(f"\n========================================================")
    print(f" Category: {category.upper()} (Total in holdout: {len(data)})")
    print(f"========================================================")

    for i, item in enumerate(data[:limit], start=1):
        print(f"\n--- Sample #{i} ---")
        print(json.dumps(item, indent=2, ensure_ascii=False))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inspect generated holdout dataset samples")
    parser.add_argument("--limit-per-cat", type=int, default=10, help="Number of samples to inspect per category (default: 10 -> 50 total)")
    parser.add_argument("--task", type=str, default="all", help="Specific task to inspect (match_scoring, dedup_verification, role_classification, section_parsing, field_mapping, or all)")
    args = parser.parse_args()

    categories = [
        "match_scoring",
        "dedup_verification",
        "role_classification",
        "section_parsing",
        "field_mapping",
    ]
    selected = categories if args.task == "all" else [args.task]
    for cat in selected:
        inspect_samples(cat, limit=args.limit_per_cat)


if __name__ == "__main__":
    main()
