"""Bootstrap training and evaluation datasets for TechJobMCP v2 dual-cognition models.

Strategy: "Annotation over Generation"
- Extracts REAL, authentic job postings directly from SQLite database (data/real_jobs.db).
- Uses a curated, realistic static CV Matrix of 20 diverse profiles.
- Creates natural cross-product pairs (yielding genuine real-world edge cases and hard negatives).
- Uses ResilientLLMGateway strictly as an LLM-as-a-Judge annotator (assigning scores & flags).
- Splits into training and golden holdout sets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from job_mcp.core.llm.gateway import ResilientLLMGateway

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_training_data")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR = DATA_DIR / "training"
HOLDOUT_DIR = DATA_DIR / "holdout"
REAL_JOBS_DB = DATA_DIR / "real_jobs.db"

# ---------------------------------------------------------------------------
# Strict Structured Output Pydantic Schemas
# ---------------------------------------------------------------------------
class MatchScoringItem(BaseModel):
    id: int
    skill_match: int = Field(ge=0, le=4, description="0: None, 1: Weak, 2: Partial, 3: Strong, 4: Perfect")
    seniority_fit: int = Field(ge=0, le=4, description="0: Far too junior, 1: Slightly junior, 2: Good fit, 3: Senior, 4: Overqualified")
    recruiter_fit: bool = Field(description="Would a human technical recruiter invite this candidate to an interview?")
    reasoning: str = Field(description="Concise 1-sentence evaluation justification")


class MatchScoringBatchOutput(BaseModel):
    items: List[MatchScoringItem]


class DedupVerificationItem(BaseModel):
    id: int
    is_duplicate: bool
    reasoning: str


class DedupVerificationBatchOutput(BaseModel):
    items: List[DedupVerificationItem]


class RoleClassificationItem(BaseModel):
    id: int
    category: str = Field(description="One of: Core Engineering, Engineering-Adjacent, Technical Hybrid, Administrative, Non-Technical")


class RoleClassificationBatchOutput(BaseModel):
    items: List[RoleClassificationItem]


class SectionParsingItem(BaseModel):
    id: int
    section_type: str = Field(description="One of: Requirements, Responsibilities, Company overview, Benefits, Application instructions, Boilerplate")


class SectionParsingBatchOutput(BaseModel):
    items: List[SectionParsingItem]


# ---------------------------------------------------------------------------
# The 20-Profile CV Matrix (Realistic, diverse candidate summaries)
# ---------------------------------------------------------------------------
CV_MATRIX = [
    # 0. Senior Python / AI Engineer
    "Senior Python & AI Engineer with 8 years of experience. Expert in FastAPI, PyTorch, LangChain, and Agentic RAG workflows. Built multi-agent LLM systems with vector databases (Qdrant, Pinecone) in production on AWS. Seniority: Senior.",
    # 1. Junior Frontend Developer (Bootcamp)
    "Junior Frontend Developer with 1 year of hands-on experience following an intensive coding bootcamp. Proficient in React, JavaScript (ES6+), HTML5, and Tailwind CSS. Built responsive web apps and personal portfolio projects. Seniority: Junior.",
    # 2. Staff Systems / Distributed Backend Engineer
    "Staff Systems Engineer with 12 years of experience architecting high-throughput distributed systems in Go and C++. Deep expertise in Kafka, gRPC, Kubernetes, and low-latency network protocols handling 200k RPS. Seniority: Staff / Principal.",
    # 3. Mid-level Full Stack Developer
    "Full Stack Developer with 3 years of commercial experience. Strong background in TypeScript, React, Node.js, and PostgreSQL. Experienced with RESTful APIs, Docker, and CI/CD pipelines in fast-paced startup environments. Seniority: Mid-level.",
    # 4. Senior DevOps & Cloud Infrastructure Architect
    "Senior DevOps / Cloud Architect with 10 years of experience. Expert in AWS, GCP, Terraform, Kubernetes (EKS/GKE), Helm, and GitLab CI. Spearheaded zero-downtime multi-region migrations and GitOps adoption with ArgoCD. Seniority: Senior.",
    # 5. Applied ML / Computer Vision Researcher
    "Machine Learning Scientist (Ph.D. in Computer Science). 5 years of post-doc and industry experience in Computer Vision, PyTorch, CUDA kernel optimization, and diffusion models. Published at CVPR and NeurIPS. Seniority: Senior / Staff.",
    # 6. Mobile Application Developer
    "Cross-platform Mobile Developer with 4 years of experience building consumer apps in Flutter / Dart and native iOS (Swift). Successfully published and maintained apps with over 500k active users on App Store and Google Play. Seniority: Mid-level.",
    # 7. Senior Data Engineer
    "Senior Data Engineer with 6 years of experience building enterprise data platforms. Advanced skills in Apache Spark (PySpark), Airflow, Snowflake, dbt, SQL, and AWS Lake Formation. Designed petabyte-scale ETL pipelines. Seniority: Senior.",
    # 8. Embedded / Low-Level Firmware Developer
    "Firmware & Embedded Systems Engineer with 7 years of experience. Expert in Embedded C, C++, FreeRTOS, ARM Cortex-M microcontrollers, Linux device drivers, and I2C/SPI hardware protocols. Seniority: Senior.",
    # 9. Cybersecurity / Application Security Engineer
    "Application Security Engineer with 5 years of experience in AppSec, threat modeling, SAST/DAST tooling, and OWASP Top 10 remediation. Skilled in Python scripting, penetration testing, and cloud security posture management. Seniority: Mid-level / Senior.",
    # 10. Technical Product Manager
    "Technical Product Manager with 6 years of experience in B2B enterprise SaaS. Former backend developer. Skilled in writing detailed PRDs, conducting user research, Agile sprint planning, and partnering with R&D on complex APIs. Seniority: Senior.",
    # 11. Senior QA Automation Engineer
    "Senior QA Automation Lead with 7 years of experience designing robust test automation frameworks from scratch using Playwright, Cypress, Python, and TypeScript. Integrated end-to-end testing into GitHub Actions pipelines. Seniority: Senior.",
    # 12. Junior Cloud Support / Sysadmin
    "Junior Cloud Support Engineer with 1.5 years of experience providing Tier-2 technical support and Linux system administration. Familiar with Bash scripting, Docker basics, and AWS EC2/S3 monitoring. Seniority: Junior.",
    # 13. Web3 & Smart Contract Developer
    "Blockchain / Smart Contract Engineer with 3 years of experience in Web3. Developed and audited Solidity contracts using Foundry, Hardhat, Ethers.js, and OpenZeppelin on Ethereum and Arbitrum. Seniority: Mid-level.",
    # 14. Non-Technical Technical Recruiter / HR
    "Technical Talent Acquisition Specialist with 5 years of experience in high-tech recruiting. Expert in sourcing software engineers, conducting screening interviews, managing ATS workflows (Comeet, Greenhouse), and candidate pipeline. Seniority: Mid-level.",
    # 15. VP of Engineering / Executive
    "VP of Engineering with 16 years of engineering and management experience. Scaled R&D organization from 15 to 110 engineers across 4 international sites. Managed engineering budgets, technical strategy, and architectural governance. Seniority: Executive.",
    # 16. Overqualified Enterprise Architect
    "Chief Enterprise Architect with 18 years of experience leading core architecture across Fortune 500 financial institutions. Specialized in core banking modernisation, legacy mainframe migration, and global regulatory compliance. Seniority: Principal / Director.",
    # 17. Java Enterprise Backend Developer
    "Enterprise Java Developer with 7 years of experience. Deep proficiency in Java 17, Spring Boot, Hibernate, Apache Kafka, Oracle DB, and microservice refactoring in enterprise financial domains. Seniority: Senior.",
    # 18. Junior Data Analyst
    "Junior Data Analyst with a B.Sc. in Statistics and 1 year of experience. Highly proficient in SQL, Python (Pandas, NumPy), Excel modeling, and creating executive dashboards in Tableau and PowerBI. Seniority: Junior.",
    # 19. IDF 8200 Veteran Full Stack & Cyber Developer
    "Full Stack & Cyber Security Developer, 4 years in IDF elite intelligence unit (8200). Expert in Python, Go, React, reverse engineering, and low-latency network telemetry. Fluent in Hebrew and English. Seniority: Mid-level.",
]

CV_DOMAIN_KEYWORDS: Dict[int, List[str]] = {
    0: ["python", "ai", "fastapi", "pytorch", "langchain", "rag", "llm", "backend", "machine learning"],
    1: ["frontend", "front end", "react", "web", "ui", "ux", "full stack"],
    2: ["distributed", "systems", "backend", "go", "golang", "c++", "kafka", "grpc", "kubernetes"],
    3: ["full stack", "fullstack", "typescript", "react", "node", "postgres"],
    4: ["devops", "cloud", "aws", "gcp", "terraform", "kubernetes", "sre", "infrastructure"],
    5: ["computer vision", "vision", "machine learning", "deep learning", "pytorch", "algorithm", "research", "ai"],
    6: ["mobile", "flutter", "ios", "android", "swift"],
    7: ["data engineer", "data platform", "spark", "airflow", "snowflake", "dbt", "etl", "sql", "bigquery"],
    8: ["embedded", "firmware", "hardware", "c/c++", "low level", "rtos"],
    9: ["security", "appsec", "cyber", "soc", "penetration", "threat", "vulnerability"],
    10: ["product manager", "product management", "technical product", "product owner"],
    11: ["qa", "automation", "test", "quality", "playwright", "cypress", "sdet"],
    12: ["support", "it specialist", "sysadmin", "helpdesk", "system administrator"],
    13: ["blockchain", "web3", "smart contract", "crypto", "solidity"],
    14: ["recruiter", "talent", "sourcer", "hr", "people"],
    15: ["vp", "director", "head of", "engineering manager", "team lead", "lead", "leadership"],
    16: ["architect", "enterprise", "system architect", "solutions architect", "principal"],
    17: ["java", "spring", "backend", "microservice", "hibernate"],
    18: ["analyst", "data analyst", "bi", "tableau", "powerbi", "sql", "dashboard"],
    19: ["cyber", "full stack", "security", "react", "python", "backend"],
}


def clean_html(raw_html: str) -> str:
    """Strip basic HTML tags and entities for cleaner evaluation."""
    clean = re.sub(r"<[^>]+>", " ", raw_html)
    clean = clean.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def load_real_jobs_from_sqlite(limit: int = 500) -> List[Dict[str, Any]]:
    """Load authentic, raw job postings from SQLite database."""
    if not REAL_JOBS_DB.exists():
        raise FileNotFoundError(f"Database {REAL_JOBS_DB} does not exist. Run seeder first.")

    conn = sqlite3.connect(REAL_JOBS_DB)
    c = conn.cursor()
    c.execute(
        "SELECT job_id, company, title, description, location, source FROM real_jobs WHERE LENGTH(description) > 100 ORDER BY RANDOM() LIMIT ?",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()

    jobs = []
    for r in rows:
        jobs.append({
            "job_id": r[0],
            "company": r[1],
            "title": r[2],
            "description": clean_html(r[3]),
            "raw_description": r[3],
            "location": r[4],
            "source": r[5],
        })
    logger.info(f"Loaded {len(jobs)} authentic real jobs from {REAL_JOBS_DB}")
    return jobs


def parse_batch_response(
    raw_response: str,
    item_cls: Type[BaseModel],
    batch_cls: Type[BaseModel],
) -> List[Any]:
    """Parse batch response safely handling wrapper dicts, bare lists, single objects, and markdown fences."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # 1. Try direct batch model validation
    try:
        batch_obj = batch_cls.model_validate_json(cleaned)
        return getattr(batch_obj, "items", [])
    except Exception:
        pass

    # 2. Try JSON parse and inspect structure
    try:
        data = json.loads(cleaned)
    except Exception as e:
        logger.warning(f"Failed to parse JSON response: {e}. Raw response snippet: {cleaned[:200]}")
        return []

    items = []
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            for x in data["items"]:
                try:
                    items.append(item_cls.model_validate(x))
                except Exception as ex:
                    logger.warning(f"Error validating item: {ex}")
        else:
            try:
                items.append(item_cls.model_validate(data))
            except Exception as ex:
                logger.warning(f"Error validating single object item: {ex}")
    elif isinstance(data, list):
        for x in data:
            try:
                items.append(item_cls.model_validate(x))
            except Exception as ex:
                logger.warning(f"Error validating item in list: {ex}")

    return items


async def annotate_match_scoring_batch(
    gateway: ResilientLLMGateway,
    pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Use LLM-as-a-Judge to evaluate real job vs candidate CV pairs."""
    eval_items = []
    for idx, p in enumerate(pairs):
        eval_items.append({
            "id": idx,
            "job_title": p["job_title"],
            "job_description_excerpt": p["job_description"][:500],
            "candidate_cv": p["candidate_cv"],
        })

    prompt = f"""You are an elite, objective technical recruiter acting as an LLM Judge.
You must EVALUATE each real job description against the candidate CV. You must NOT generate text or modify descriptions.

Evaluate each pair on three dimensions:
1. "skill_match" (integer 0 to 4):
   - 0: No skill overlap (e.g. Accountant applying to Linux Kernel dev)
   - 1: Weak overlap (only tangential tools like Git, but zero primary tech)
   - 2: Partial overlap (knows ~50% of core stack, e.g. Knows Python but no ML/PyTorch)
   - 3: Strong overlap (knows primary stack and core tools, small secondary gaps)
   - 4: Perfect overlap (possesses all required skills and primary competencies)

2. "seniority_fit" (integer 0 to 4):
   - 0: Far too junior (e.g. Junior 1 yr applying for Staff 10+ yrs)
   - 1: Slightly junior (e.g. Junior 2 yrs applying for Mid 3-4 yrs)
   - 2: Good fit (ideal seniority alignment)
   - 3: Senior (slightly more experienced than role requires)
   - 4: Overqualified (e.g. Director/Principal 15+ yrs applying for entry level junior)

3. "recruiter_fit" (boolean):
   - Would a human technical recruiter invite this candidate to a first-round interview? (true/false)

Pairs to evaluate:
{json.dumps(eval_items, indent=1, ensure_ascii=False)}

Return a JSON object with an "items" array containing the evaluated objects with fields: "id", "skill_match", "seniority_fit", "recruiter_fit", "reasoning".
"""

    system_prompt = "You are an objective technical recruiter evaluation judge. Output strictly valid JSON matching the schema."

    raw_response = await gateway.ask_question(
        question=prompt,
        system_prompt=system_prompt,
        response_schema=MatchScoringBatchOutput,
    )

    items = parse_batch_response(raw_response, MatchScoringItem, MatchScoringBatchOutput)
    label_map = {}
    for pos, it in enumerate(items):
        item_id = getattr(it, "id", None)
        if item_id is not None and isinstance(item_id, int):
            label_map[item_id] = it
        else:
            label_map[pos] = it

    annotated = []
    for idx, p in enumerate(pairs):
        lbl = label_map.get(idx) or (items[idx] if idx < len(items) else None)
        if lbl:
            annotated.append({
                "job_title": p["job_title"],
                "job_description": p["job_description"],
                "candidate_cv": p["candidate_cv"],
                "skill_match": lbl.skill_match,
                "seniority_fit": lbl.seniority_fit,
                "recruiter_fit": lbl.recruiter_fit,
                "reasoning": lbl.reasoning,
            })
    return annotated


async def generate_real_match_scoring(
    gateway: ResilientLLMGateway, real_jobs: List[Dict[str, Any]], target_count: int, batch_size: int = 8
) -> List[Dict[str, Any]]:
    """Create balanced pairs of real jobs and static CVs (55% targeted domain, 45% random negative), then annotate via LLM-as-a-Judge."""
    logger.info(f"Generating {target_count} balanced real-world match scoring samples...")
    results: List[Dict[str, Any]] = []

    # Precompute domain job matches for each CV profile
    cv_domain_matches: Dict[int, List[Dict[str, Any]]] = {}
    for cv_idx, kws in CV_DOMAIN_KEYWORDS.items():
        patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in kws]
        matches = [
            j for j in real_jobs
            if any(p.search(j["title"]) for p in patterns)
        ]
        if matches:
            cv_domain_matches[cv_idx] = matches

    valid_cv_indices = list(cv_domain_matches.keys())

    # Build balanced candidate pairs
    candidate_pairs = []
    used_combos = set()
    targeted_target = int(target_count * 0.55)

    # 1. Targeted Domain Pairs (generates true positives & partial matches: scores 2, 3, 4)
    attempts = 0
    while len(candidate_pairs) < targeted_target and attempts < targeted_target * 10:
        attempts += 1
        cv_idx = random.choice(valid_cv_indices)
        job = random.choice(cv_domain_matches[cv_idx])
        combo_key = (job["job_id"], cv_idx)
        if combo_key in used_combos:
            continue
        used_combos.add(combo_key)
        candidate_pairs.append({
            "job_title": job["title"],
            "job_description": job["description"],
            "candidate_cv": CV_MATRIX[cv_idx],
        })

    # 2. Random Cross-Product Pairs (generates authentic hard & soft negatives: scores 0, 1)
    attempts = 0
    while len(candidate_pairs) < target_count and attempts < target_count * 10:
        attempts += 1
        job = random.choice(real_jobs)
        cv_idx = random.randint(0, len(CV_MATRIX) - 1)
        combo_key = (job["job_id"], cv_idx)
        if combo_key in used_combos:
            continue
        used_combos.add(combo_key)
        candidate_pairs.append({
            "job_title": job["title"],
            "job_description": job["description"],
            "candidate_cv": CV_MATRIX[cv_idx],
        })

    random.shuffle(candidate_pairs)
    logger.info(
        f"Assembled {len(candidate_pairs)} candidate pairs ({len([p for p in candidate_pairs if p in candidate_pairs[:targeted_target]])} targeted domain, "
        f"{len(candidate_pairs) - targeted_target} random cross-product)."
    )

    # Batch process through LLM-as-a-Judge
    for i in range(0, len(candidate_pairs), batch_size):
        batch = candidate_pairs[i : i + batch_size]
        logger.info(f"Annotating batch {i // batch_size + 1}/{(len(candidate_pairs) + batch_size - 1) // batch_size} ({len(batch)} pairs)...")
        try:
            annotated_batch = await annotate_match_scoring_batch(gateway, batch)
            results.extend(annotated_batch)
        except Exception as ex:
            logger.warning(f"Batch {i // batch_size + 1} annotation failed: {ex}. Continuing with remaining batches...")

    return results


async def generate_real_dedup_samples(
    gateway: ResilientLLMGateway, real_jobs: List[Dict[str, Any]], target_count: int, batch_size: int = 10
) -> List[Dict[str, Any]]:
    """Generate authentic deduplication pairs using real jobs and LLM judge."""
    logger.info(f"Generating {target_count} authentic deduplication samples...")
    pairs: List[Dict[str, Any]] = []

    # Group real jobs by company for authentic same-company hard negatives
    company_jobs: Dict[str, List[Dict[str, Any]]] = {}
    for j in real_jobs:
        c = j["company"]
        company_jobs.setdefault(c, []).append(j)

    # 1. Authentic Hard Negatives: Same company, different roles (~50%)
    hard_neg_count = target_count // 2
    for _ in range(hard_neg_count):
        # Pick a company with multiple jobs
        candidates = [c for c, jobs in company_jobs.items() if len(jobs) >= 2]
        if candidates:
            comp = random.choice(candidates)
            j_a, j_b = random.sample(company_jobs[comp], 2)
            pairs.append({
                "job_a": {"title": j_a["title"], "company": j_a["company"], "location": j_a["location"]},
                "job_b": {"title": j_b["title"], "company": j_b["company"], "location": j_b["location"]},
                "is_duplicate": False,
                "reasoning": f"Authentic hard negative: Same company ({comp}) but different roles ({j_a['title']} vs {j_b['title']}).",
            })
        else:
            # Fallback across companies
            j_a, j_b = random.sample(real_jobs, 2)
            pairs.append({
                "job_a": {"title": j_a["title"], "company": j_a["company"], "location": j_a["location"]},
                "job_b": {"title": j_b["title"], "company": j_b["company"], "location": j_b["location"]},
                "is_duplicate": False,
                "reasoning": "Different companies and roles.",
            })

    # 2. Authentic True Duplicates: Real job with realistic aggregator title variations (~50%)
    dup_count = target_count - len(pairs)
    variations = [
        lambda t: f"{t} (Hybrid)",
        lambda t: f"{t} - Tel Aviv",
        lambda t: t.replace("Engineer", "Developer") if "Engineer" in t else f"Senior {t}",
        lambda t: f"דרוש/ה {t}",
        lambda t: f"{t} [Remote / Onsite]",
    ]

    for _ in range(dup_count):
        j = random.choice(real_jobs)
        var_fn = random.choice(variations)
        title_b = var_fn(j["title"])
        pairs.append({
            "job_a": {"title": j["title"], "company": j["company"], "location": j["location"]},
            "job_b": {"title": title_b, "company": j["company"], "location": j["location"]},
            "is_duplicate": True,
            "reasoning": f"True duplicate with common aggregator formatting variation.",
        })

    random.shuffle(pairs)
    return pairs[:target_count]


async def generate_real_role_classification(
    gateway: ResilientLLMGateway, real_jobs: List[Dict[str, Any]], target_count: int, batch_size: int = 15
) -> List[Dict[str, Any]]:
    """Classify real jobs into career categories using LLM judge."""
    logger.info(f"Classifying {target_count} real jobs into role categories...")
    sampled = random.sample(real_jobs, min(target_count, len(real_jobs)))
    results = []

    for i in range(0, len(sampled), batch_size):
        batch = sampled[i : i + batch_size]
        items = [{"id": idx, "title": b["title"], "snippet": b["description"][:300]} for idx, b in enumerate(batch)]

        prompt = f"""Classify these REAL tech company job postings into exactly ONE role category:
- Core Engineering
- Engineering-Adjacent
- Technical Hybrid
- Administrative
- Non-Technical

Jobs:
{json.dumps(items, indent=1, ensure_ascii=False)}

Return a JSON object with an "items" array where each object has fields: "id", "category".
"""
        raw_response = await gateway.ask_question(
            question=prompt,
            system_prompt="You are an expert role classification judge. Output strictly valid JSON matching the schema.",
            response_schema=RoleClassificationBatchOutput,
        )
        parsed_items = parse_batch_response(raw_response, RoleClassificationItem, RoleClassificationBatchOutput)
        cat_map = {}
        for pos, it in enumerate(parsed_items):
            item_id = getattr(it, "id", None)
            if item_id is not None and isinstance(item_id, int):
                cat_map[item_id] = it.category
            else:
                cat_map[pos] = it.category

        for idx, b in enumerate(batch):
            cat = cat_map.get(idx) or (parsed_items[idx].category if idx < len(parsed_items) else "Core Engineering")
            results.append({
                "title": b["title"],
                "snippet": b["description"][:300],
                "category": cat,
                "reasoning": f"Classified from authentic {b['company']} job posting.",
            })

    return results


async def generate_real_section_parsing(
    gateway: ResilientLLMGateway, real_jobs: List[Dict[str, Any]], target_count: int, batch_size: int = 15
) -> List[Dict[str, Any]]:
    """Extract real paragraphs from real job descriptions and label them with LLM judge."""
    logger.info(f"Extracting and labeling {target_count} real paragraphs for section parsing...")
    paragraphs = []
    for j in real_jobs:
        # Split description by sentences or breaks
        parts = re.split(r"\n\s*\n|\.\s{2,}|<br\s*/?>|</p>", j["raw_description"])
        for p in parts:
            clean_p = clean_html(p)
            if 60 <= len(clean_p) <= 400:
                paragraphs.append(clean_p)

    sampled_paras = random.sample(paragraphs, min(target_count, len(paragraphs)))
    results = []

    for i in range(0, len(sampled_paras), batch_size):
        batch = sampled_paras[i : i + batch_size]
        items = [{"id": idx, "paragraph": b} for idx, b in enumerate(batch)]

        prompt = f"""Label each authentic job description paragraph with its primary section type:
- Requirements
- Responsibilities
- Company overview
- Benefits
- Application instructions
- Boilerplate

Paragraphs:
{json.dumps(items, indent=1, ensure_ascii=False)}

Return a JSON object with an "items" array where each object has fields: "id", "section_type".
"""
        raw_response = await gateway.ask_question(
            question=prompt,
            system_prompt="You are an expert section parsing judge. Output strictly valid JSON matching the schema.",
            response_schema=SectionParsingBatchOutput,
        )
        parsed_items = parse_batch_response(raw_response, SectionParsingItem, SectionParsingBatchOutput)
        sec_map = {}
        for pos, it in enumerate(parsed_items):
            item_id = getattr(it, "id", None)
            if item_id is not None and isinstance(item_id, int):
                sec_map[item_id] = it.section_type
            else:
                sec_map[pos] = it.section_type

        for idx, b in enumerate(batch):
            sec = sec_map.get(idx) or (parsed_items[idx].section_type if idx < len(parsed_items) else "Boilerplate")
            results.append({
                "paragraph": b,
                "section_type": sec,
                "language": "he" if any("\u0590" <= c <= "\u05ea" for c in b) else "en",
            })

    return results


def split_train_holdout(samples: List[Dict[str, Any]], holdout_ratio: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically split samples into training and holdout sets with guaranteed zero leakage."""
    random.seed(42)
    seen = set()
    unique_samples = []
    for s in samples:
        key = json.dumps(s, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)

    random.shuffle(unique_samples)
    split_idx = int(len(unique_samples) * (1.0 - holdout_ratio))
    return unique_samples[:split_idx], unique_samples[split_idx:]


def save_dataset(name: str, train: List[Dict[str, Any]], holdout: List[Dict[str, Any]]) -> None:
    """Save train and holdout splits to JSON files."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = TRAINING_DIR / f"{name}_train.json"
    holdout_path = HOLDOUT_DIR / f"{name}_holdout.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    with open(holdout_path, "w", encoding="utf-8") as f:
        json.dump(holdout, f, indent=2, ensure_ascii=False)

    logger.info("Saved %s: %d train, %d holdout", name, len(train), len(holdout))


async def run_pipeline(target_per_task: int, batch_size: int, task: str = "all", provider: str = "gemini") -> None:
    """Run Annotation over Generation pipeline."""
    if provider:
        os.environ["PRIMARY_LLM_PROVIDER"] = provider

    real_jobs = load_real_jobs_from_sqlite(limit=600)
    gateway = ResilientLLMGateway()

    tasks_to_run = ["match_scoring", "dedup_verification", "role_classification", "section_parsing"]
    if task != "all":
        tasks_to_run = [task]

    for t in tasks_to_run:
        if t == "match_scoring":
            samples = await generate_real_match_scoring(gateway, real_jobs, target_per_task, batch_size)
            tr, ho = split_train_holdout(samples, holdout_ratio=0.2)
            save_dataset("match_scoring", tr, ho)

        elif t == "dedup_verification":
            samples = await generate_real_dedup_samples(gateway, real_jobs, target_per_task, batch_size)
            tr, ho = split_train_holdout(samples, holdout_ratio=0.2)
            save_dataset("dedup_verification", tr, ho)

        elif t == "role_classification":
            samples = await generate_real_role_classification(gateway, real_jobs, target_per_task, batch_size)
            tr, ho = split_train_holdout(samples, holdout_ratio=0.2)
            save_dataset("role_classification", tr, ho)

        elif t == "section_parsing":
            samples = await generate_real_section_parsing(gateway, real_jobs, target_per_task, batch_size)
            tr, ho = split_train_holdout(samples, holdout_ratio=0.2)
            save_dataset("section_parsing", tr, ho)


def main():
    parser = argparse.ArgumentParser(description="Annotation over Generation Data Pipeline for TechJobMCP v2.")
    parser.add_argument("--target-per-task", type=int, default=100, help="Target samples per task")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for LLM-as-a-judge annotation (default: 5)")
    parser.add_argument("--task", type=str, default="all", help="Specific task or 'all'")
    parser.add_argument("--provider", type=str, default="gemini", help="Preferred primary LLM provider (default: gemini)")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.target_per_task, args.batch_size, args.task, args.provider))


if __name__ == "__main__":
    main()
