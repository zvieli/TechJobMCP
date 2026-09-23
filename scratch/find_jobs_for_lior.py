import asyncio
import time
from job_mcp.core.api_client import extract_candidate_profile, filter_jobs
from job_mcp.core.system1.engine import LazyLayaEngine
from job_mcp.models.schemas import JobPreferences, WorkMode
from job_mcp.sources.public.comeet import ComeetSource
from job_mcp.sources.public.greenhouse import GreenhouseSource
from job_mcp.sources.enterprise.direct_tech import DirectTechSource
from job_mcp.sources.dedup import deduplicate_jobs

async def main():
    print("=" * 85)
    print(" 🚀 LIVE JOB MATCHING WITH SYSTEM 1 NEURAL ENSEMBLE FOR LIOR ZVIELI")
    print("=" * 85)

    # 1. Warm up System 1 in RAM
    engine = LazyLayaEngine.get_instance()
    t0_warm = time.perf_counter()
    engine.warmup()
    print(f"✅ System 1 Neural Engine (LazyLayaEngine) active in RAM ({(time.perf_counter()-t0_warm)*1000:.1f}ms)")

    # 2. Extract CV profile
    cv_path = "cv.pdf"
    profile = extract_candidate_profile(cv_path)
    print(f"\n📄 Profile Grounding ({cv_path}):")
    print(f"   Candidate: Lior Zvieli")
    print(f"   Target Roles: {profile.target_roles[:4]}")
    print(f"   Primary Tech Stack: {profile.primary_stack[:6]}")
    print(f"   Inferred Seniority: {profile.seniority_level or 'Junior / Entry-Level Engineer'}")

    # 3. Fetch from live sources
    print("\n🌐 Fetching live openings from ATS sources (Comeet, Greenhouse, Direct Tech)...")
    t0_fetch = time.perf_counter()
    c_source = ComeetSource()
    g_source = GreenhouseSource()
    d_source = DirectTechSource()

    c_jobs, g_jobs, d_jobs = await asyncio.gather(
        c_source.fetch_jobs(limit=500),
        g_source.fetch_jobs(limit=500),
        d_source.fetch_jobs(limit=100),
        return_exceptions=True,
    )
    
    raw_jobs = []
    if isinstance(c_jobs, list):
        print(f"   • Comeet: {len(c_jobs)} live jobs")
        raw_jobs.extend(c_jobs)
    if isinstance(g_jobs, list):
        print(f"   • Greenhouse: {len(g_jobs)} live jobs")
        raw_jobs.extend(g_jobs)
    if isinstance(d_jobs, list):
        print(f"   • Direct Tech: {len(d_jobs)} live jobs")
        raw_jobs.extend(d_jobs)

    fetch_duration = time.perf_counter() - t0_fetch
    deduped_jobs = deduplicate_jobs(raw_jobs)
    print(f"✅ Fetched {len(raw_jobs)} total jobs, deduplicated to {len(deduped_jobs)} unique listings in {fetch_duration:.2f}s.")

    # 4. Filter & Score with System 1
    prefs = JobPreferences(
        cv_path=cv_path,
        location="Israel",
        keywords=["AI", "Python", "Full Stack", "Backend", "Software", "Machine Learning"],
        exclude_keywords=["Principal", "Staff", "Director", "VP", "Head", "Architect", "10+ years"],
    )

    print("\n🧠 Evaluating jobs with System 1 Neural Ensemble (Vectorized CPU Chunking)...")
    t0_score = time.perf_counter()
    scored_jobs = filter_jobs(deduped_jobs, prefs, profile=profile, enable_system1=True)
    score_duration = time.perf_counter() - t0_score
    print(f"✅ Scored and ranked {len(scored_jobs)} candidate jobs in {score_duration:.2f}s.")

    # 5. Display Top Matches
    print("\n" + "=" * 85)
    print(" 🎯 TOP MATCHED JOB OPPORTUNITIES FOR LIOR")
    print("=" * 85)

    for i, j in enumerate(scored_jobs[:10], 1):
        score = j.match_score if j.match_score is not None else 0.0
        conf = j.system1_confidence
        conf_str = f"{conf:.1%}" if conf is not None else "N/A"
        req_s2 = j.requires_system2_review
        status = "✅ Auto-Triaged (High Confidence)" if not req_s2 else "⚠️ Flagged for System 2 Review"
        wm_str = j.work_mode.value if isinstance(j.work_mode, WorkMode) else str(j.work_mode or "Not specified")

        print(f"\n#{i} | {j.title} @ {j.company}")
        print(f"    📍 Location: {j.location} | Work Mode: {wm_str} | ATS Source: {j.source}")
        print(f"    ⭐ System 1 Match Score: {score:.1f}/100.0 | Confidence: {conf_str} ({status})")
        print(f"    🎖️ Seniority Level: {j.seniority_level or 'Junior / Mid / Not specified'}")
        if j.matched_skills:
            print(f"    🛠️ Core Matched Skills: {', '.join(j.matched_skills[:8])}")
        print(f"    💡 Match Explanations:")
        for r in (j.match_reasons or [])[:3]:
            print(f"       • {r}")
        if j.url:
            print(f"    🔗 Direct Apply Link: {j.url}")

    # 6. Seniority Calibration Verification Section
    print("\n" + "=" * 85)
    print(" 🛡️ SENIORITY CALIBRATION CHECK (Zero False Positives on Senior Roles)")
    print("=" * 85)
    test_senior_titles = ["Senior Full Stack Engineer", "Senior Backend Engineer", "Lead", "Director"]
    found_seniors = [j for j in scored_jobs if any(st.lower() in j.title.lower() for st in test_senior_titles)][:5]
    if not found_seniors:
        # Check from all deduped to inspect how they were scored
        from job_mcp.core.api_client import calculate_match_score
        found_seniors = [j for j in deduped_jobs if any(st.lower() in j.title.lower() for st in test_senior_titles)][:5]
        for sj in found_seniors:
            calculate_match_score(sj, prefs, profile=profile, enable_system1=True)

    for sj in found_seniors:
        print(f" • {sj.title} @ {sj.company}")
        print(f"   Score: {sj.match_score:.1f}/100.0 | Confidence: {sj.system1_confidence:.1%} | Requires S2: {sj.requires_system2_review}")

if __name__ == "__main__":
    asyncio.run(main())
