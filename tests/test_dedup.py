"""Tests for Data Models Extension & Deduplication / Entity Merger Engine."""

import pytest
from job_mcp.models.schemas import Job, WorkMode
from job_mcp.sources.dedup import (
    normalize_title,
    normalize_company,
    compute_dedup_key,
    merge_job_entities,
    deduplicate_jobs,
)


class TestNormalizeTitle:
    """Test suite for title normalization."""

    def test_lowercase_and_strip_punctuation(self):
        assert normalize_title("Senior Python Developer!") == "senior python developer"
        assert normalize_title("Backend Engineer / Architect") == "backend engineer architect"

    def test_strip_gender_annotations_english(self):
        assert normalize_title("Senior Backend Engineer (m/f/d)") == "senior backend engineer"
        assert normalize_title("Full Stack Developer [f/m]") == "full stack developer"
        assert normalize_title("DevOps Engineer (m/w/d)") == "devops engineer"
        assert normalize_title("Product Manager m/f") == "product manager"

    def test_strip_gender_annotations_hebrew(self):
        assert normalize_title("מפתח/ת תוכנה (ז/נ)") == "מפתח תוכנה"
        assert normalize_title("מהנדס/ת DevOps (נ/ז)") == "מהנדס devops"
        assert normalize_title("ראש/ת צוות פיתוח") == "ראש צוות פיתוח"
        assert normalize_title("איש/אשת סיסטם") == "איש סיסטם"

    def test_strip_work_mode_noise(self):
        assert normalize_title("Data Scientist (Hybrid)") == "data scientist"
        assert normalize_title("Frontend Engineer (Remote)") == "frontend engineer"
        assert normalize_title("QA Automation Engineer (Onsite)") == "qa automation engineer"
        assert normalize_title("מפתח Python (היברידי)") == "מפתח python"
        assert normalize_title("ארכיטקט ענן (מרחוק)") == "ארכיטקט ענן"

    def test_strip_filler_keywords(self):
        assert normalize_title("[Urgent] DevOps Engineer - We are hiring") == "devops engineer"
        assert normalize_title("דרוש/ה מפתח/ת Fullstack (משרה מלאה)") == "מפתח fullstack"
        assert normalize_title("Senior Java Developer - Immediate opening") == "senior java developer"

    def test_preserve_tech_keywords(self):
        assert "c++" in normalize_title("Senior C++ Developer")
        assert "c#" in normalize_title("C# / .NET Backend Engineer") or "csharp" in normalize_title("C# / .NET Backend Engineer")

    def test_empty_and_whitespace(self):
        assert normalize_title("") == ""
        assert normalize_title("   ") == ""


class TestNormalizeCompany:
    """Test suite for company normalization."""

    def test_strip_corporate_suffixes_english(self):
        assert normalize_company("Google LLC") == "google"
        assert normalize_company("Wix.com Ltd.") == "wix"
        assert normalize_company("Meta Platforms Inc.") == "meta platforms"
        assert normalize_company("Amazon Web Services Corp") == "amazon web services"

    def test_strip_corporate_suffixes_hebrew(self):
        assert normalize_company("אלביט מערכות בע\"מ") == "אלביט מערכות"
        assert normalize_company("רפאל מערכות בעמ") == "רפאל מערכות"
        res = normalize_company("צ'ק פוינט טכנולוגיות בע מ")
        assert "פוינט" in res and "טכנולוגיות" not in res and "בע" not in res

    def test_strip_domain_extensions(self):
        assert normalize_company("Monday.com") == "monday"
        assert normalize_company("Fiverr.com Ltd") == "fiverr"
        assert normalize_company("Lemonade.io") == "lemonade"

    def test_strip_technologies_and_tech_suffixes(self):
        assert normalize_company("Check Point Software Technologies Ltd") == "check point software" or normalize_company("Check Point Software Technologies Ltd") == "check point"
        assert normalize_company("SolarEdge Technologies") == "solaredge"

    def test_fallback_for_suffix_only_name(self):
        assert normalize_company("Tech Corp") == "tech"
        assert normalize_company("Group Ltd") == "group"

    def test_empty_and_whitespace(self):
        assert normalize_company("") == ""
        assert normalize_company("   ") == ""


class TestComputeDedupKey:
    """Test suite for deduplication key generation."""

    def test_matching_keys_across_variations(self):
        key1 = compute_dedup_key("Senior Python Engineer (m/f/d)", "Acme Inc.")
        key2 = compute_dedup_key("Senior Python Engineer (Hybrid)", "Acme Ltd.")
        assert key1 == key2
        assert key1 == "senior python engineer@acme"

    def test_matching_keys_hebrew_english_variations(self):
        key1 = compute_dedup_key("מפתח/ת Python (היברידי)", "אלביט בע\"מ")
        key2 = compute_dedup_key("מפתח Python", "אלביט בעמ")
        assert key1 == key2
        assert key1 == "מפתח python@אלביט"

    def test_distinct_jobs_different_keys(self):
        key_dev = compute_dedup_key("Senior Python Developer", "Google LLC")
        key_qa = compute_dedup_key("Senior QA Engineer", "Google LLC")
        key_other_comp = compute_dedup_key("Senior Python Developer", "Meta Inc")
        assert key_dev != key_qa
        assert key_dev != key_other_comp


class TestJobSchemaExtension:
    """Test suite for Job model backward-compatible schema extensions."""

    def test_job_default_fields(self):
        job = Job(
            job_id="test-1",
            title="Software Engineer",
            company="Acme Corp",
        )
        assert job.source == "hiremetech"
        assert job.sources == ["hiremetech"]
        assert job.apply_url is None
        assert job.department is None

    def test_job_custom_source_and_fields(self):
        job = Job(
            job_id="comeet-123",
            title="Staff Engineer",
            company="Wix",
            source="comeet",
            sources=["comeet"],
            apply_url="https://app.comeet.com/jobs/wix/123",
            department="Core Platform",
        )
        assert job.source == "comeet"
        assert job.sources == ["comeet"]
        assert job.apply_url == "https://app.comeet.com/jobs/wix/123"
        assert job.department == "Core Platform"

    def test_job_auto_syncs_source_to_sources(self):
        job = Job(
            job_id="alljobs-456",
            title="DevOps Lead",
            company="Monday",
            source="alljobs",
        )
        assert job.source == "alljobs"
        assert "alljobs" in job.sources


class TestMergeJobEntities:
    """Test suite for merging two Job entity models."""

    def test_merge_sources_union(self):
        job1 = Job(
            job_id="hmt-1",
            title="Backend Engineer",
            company="Acme",
            source="hiremetech",
            sources=["hiremetech"],
        )
        job2 = Job(
            job_id="cmt-2",
            title="Backend Engineer",
            company="Acme",
            source="comeet",
            sources=["comeet"],
        )
        merged = merge_job_entities(job1, job2)
        assert "hiremetech" in merged.sources
        assert "comeet" in merged.sources
        assert len(merged.sources) == 2

    def test_merge_ats_apply_url_priority(self):
        # comeet.com > hiremetech.com > alljobs.co.il
        job_alljobs = Job(
            job_id="aj-1",
            title="Python Developer",
            company="Acme",
            source="alljobs",
            url="https://alljobs.co.il/jobs/123",
            apply_url="https://alljobs.co.il/apply/123",
        )
        job_hmt = Job(
            job_id="hmt-1",
            title="Python Developer",
            company="Acme",
            source="hiremetech",
            url="https://hiremetech.com/jobs/456",
            apply_url="https://hiremetech.com/apply/456",
        )
        job_comeet = Job(
            job_id="cmt-1",
            title="Python Developer",
            company="Acme",
            source="comeet",
            url="https://app.comeet.com/jobs/acme/789",
            apply_url="https://app.comeet.com/jobs/acme/789/apply",
        )

        merged_hmt_aj = merge_job_entities(job_alljobs, job_hmt)
        assert "hiremetech.com" in (merged_hmt_aj.apply_url or "") or "hiremetech.com" in (merged_hmt_aj.url or "")

        merged_ats = merge_job_entities(job_hmt, job_comeet)
        assert "comeet.com" in (merged_ats.apply_url or "")

    def test_merge_tech_stack_union(self):
        job1 = Job(
            job_id="1",
            title="Dev",
            company="Acme",
            tech_stack=["Python", "FastAPI"],
        )
        job2 = Job(
            job_id="2",
            title="Dev",
            company="Acme",
            tech_stack=["FastAPI", "Docker", "AWS"],
        )
        merged = merge_job_entities(job1, job2)
        assert set(merged.tech_stack) == {"Python", "FastAPI", "Docker", "AWS"}
        assert len(merged.tech_stack) == 4

    def test_merge_description_prefers_longer(self):
        job1 = Job(
            job_id="1",
            title="Dev",
            company="Acme",
            description="Short snippet description.",
        )
        job2 = Job(
            job_id="2",
            title="Dev",
            company="Acme",
            description="Comprehensive description with responsibilities, tech stack, and full requirements list.",
        )
        merged = merge_job_entities(job1, job2)
        assert merged.description == job2.description

    def test_merge_preserves_best_available_metadata(self):
        job1 = Job(
            job_id="1",
            title="Dev",
            company="Acme",
            salary_range=None,
            work_mode=WorkMode.REMOTE,
            department=None,
            is_bookmarked=False,
            match_score=85.0,
            posted_date="2026-08-15",
        )
        job2 = Job(
            job_id="2",
            title="Dev",
            company="Acme",
            salary_range="35,000 - 45,000 ILS",
            work_mode=None,
            department="Core Infrastructure",
            is_bookmarked=True,
            match_score=92.5,
            posted_date=None,
        )
        merged = merge_job_entities(job1, job2)
        assert merged.salary_range == "35,000 - 45,000 ILS"
        assert merged.work_mode == WorkMode.REMOTE
        assert merged.department == "Core Infrastructure"
        assert merged.is_bookmarked is True
        assert merged.match_score == 92.5
        assert merged.posted_date == "2026-08-15"


class TestDeduplicateJobs:
    """Test suite for full list deduplication."""

    def test_deduplicate_empty_list(self):
        assert deduplicate_jobs([]) == []

    def test_deduplicate_single_job(self):
        job = Job(job_id="1", title="Dev", company="Acme")
        assert deduplicate_jobs([job]) == [job]

    def test_deduplicate_multiple_sources_merging(self):
        job_hmt = Job(
            job_id="hmt-1",
            title="Senior Python Engineer (m/f/d)",
            company="Acme Ltd",
            source="hiremetech",
            tech_stack=["Python", "FastAPI"],
            description="Short snippet.",
            url="https://hiremetech.com/jobs/1",
        )
        job_comeet = Job(
            job_id="cmt-1",
            title="Senior Python Engineer (Hybrid)",
            company="Acme Inc.",
            source="comeet",
            tech_stack=["Python", "Docker", "PostgreSQL"],
            description="Full long description with requirements.",
            apply_url="https://app.comeet.com/jobs/acme/1",
        )
        job_other = Job(
            job_id="hmt-2",
            title="Frontend Specialist",
            company="Beta Corp",
            source="hiremetech",
            tech_stack=["React", "TypeScript"],
        )

        raw_jobs = [job_hmt, job_other, job_comeet]
        deduped = deduplicate_jobs(raw_jobs)

        assert len(deduped) == 2
        # First job is the merged Acme Python Engineer
        acme_job = next(j for j in deduped if "acme" in j.company.lower())
        assert set(acme_job.sources) == {"hiremetech", "comeet"}
        assert set(acme_job.tech_stack) == {"Python", "FastAPI", "Docker", "PostgreSQL"}
        assert acme_job.description == "Full long description with requirements."
        assert acme_job.apply_url == "https://app.comeet.com/jobs/acme/1"

        # Second job is Beta Corp frontend
        beta_job = next(j for j in deduped if "beta" in j.company.lower())
        assert beta_job.job_id == "hmt-2"
        assert beta_job.sources == ["hiremetech"]
