"""Unit and integration tests for SemanticFormMapper and ATS form resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from job_mcp.core.application.mapper import SemanticFormMapper
from job_mcp.core.application.strategies.api import ApiPostStrategy
from job_mcp.core.application.strategies.browser import BrowserPlaywrightStrategy
from job_mcp.core.llm.cache import LLMCache
from job_mcp.core.llm.gateway import ResilientLLMGateway
from job_mcp.core.llm.rate_limiter import TokenBucketRateLimiter
from job_mcp.models.schemas import CandidateProfile, Job, WorkMode


@pytest.fixture
def mock_llm_gateway() -> MagicMock:
    """Fixture providing a mock ResilientLLMGateway with spy on ask_question."""
    gateway = MagicMock(spec=ResilientLLMGateway)
    gateway.ask_question = AsyncMock(return_value="Answer from LLM gateway.")
    return gateway


@pytest.fixture
def live_memory_gateway() -> ResilientLLMGateway:
    """Fixture providing a real ResilientLLMGateway with in-memory SQLite cache and fast rate limiter."""
    cache = LLMCache(db_path=":memory:")
    limiter = TokenBucketRateLimiter(rpm=600)
    return ResilientLLMGateway(
        cache=cache,
        rate_limiter=limiter,
        initial_backoff=0.01,
        mock_fallback=True,
    )


@pytest.fixture
def sample_profile() -> CandidateProfile:
    """Fixture providing a rich CandidateProfile instance."""
    return CandidateProfile(
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AsyncIO", "Kubernetes"],
        top_skills=["Python", "FastAPI", "PostgreSQL"],
        primary_stack=["Python", "PostgreSQL"],
        seniority_level="Senior",
        target_roles=["Senior Python Engineer", "Backend Tech Lead"],
    )


@pytest.fixture
def sample_profile_dict() -> dict:
    """Fixture providing a complete profile dictionary with personal details."""
    return {
        "first_name": "Lior",
        "last_name": "Zvieli",
        "full_name": "Lior Zvieli",
        "email": "lior@example.com",
        "phone": "+972-54-1234567",
        "linkedin": "https://www.linkedin.com/in/liorzvieli",
        "github": "https://github.com/zvieli",
        "portfolio": "https://zvieli.dev",
        "location": "Tel Aviv, Israel",
        "current_title": "Lead Software Architect",
        "current_company": "CloudTech Systems",
        "cv_path": "/home/lior/cv.pdf",
        "seniority_level": "Senior",
        "skills": ["Python", "FastAPI", "PostgreSQL"],
    }


# ===========================================================================
# 1. Zero-Cost Regex Heuristics for Standard Personal / Contact Fields
# ===========================================================================

@pytest.mark.asyncio
async def test_standard_fields_bypass_llm_with_profile_dict(
    mock_llm_gateway: MagicMock, sample_profile_dict: dict
) -> None:
    """Verify standard personal/contact fields match regex heuristics with ZERO LLM calls."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    test_fields = [
        ("first_name", "First Name", "text", "Lior"),
        ("fname", "Given Name", "text", "Lior"),
        ("last_name", "Last Name", "text", "Zvieli"),
        ("lname", "Surname / Family Name", "text", "Zvieli"),
        ("full_name", "Full Name", "text", "Lior Zvieli"),
        ("applicant_name", "Candidate Name", "text", "Lior Zvieli"),
        ("email", "Email Address", "email", "lior@example.com"),
        ("applicant_email", "Primary E-mail", "email", "lior@example.com"),
        ("phone", "Phone Number", "tel", "+972-54-1234567"),
        ("mobile_number", "Mobile / Cell", "tel", "+972-54-1234567"),
        ("linkedin_url", "LinkedIn Profile URL", "url", "https://www.linkedin.com/in/liorzvieli"),
        ("github_url", "GitHub Profile", "url", "https://github.com/zvieli"),
        ("portfolio_url", "Personal Website / Portfolio", "url", "https://zvieli.dev"),
        ("location", "Current City / Location", "text", "Tel Aviv, Israel"),
        ("current_title", "Current Job Title / Role", "text", "Lead Software Architect"),
        ("current_company", "Current Employer / Company", "text", "CloudTech Systems"),
        ("resume_file", "Upload Resume / CV", "file", "/home/lior/cv.pdf"),
    ]

    for field_id, label, ftype, expected_val in test_fields:
        res = await mapper.resolve_field(
            field_id=field_id,
            label=label,
            field_type=ftype,
            profile=sample_profile_dict,
        )
        assert res == expected_val, f"Failed resolving {field_id} ({label})"

    # Critical requirement: Zero LLM calls should have occurred!
    mock_llm_gateway.ask_question.assert_not_called()


@pytest.mark.asyncio
async def test_standard_fields_with_candidate_profile_model(
    mock_llm_gateway: MagicMock, sample_profile: CandidateProfile
) -> None:
    """Verify CandidateProfile Pydantic model attributes map correctly without calling LLM."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    title_res = await mapper.resolve_field(
        field_id="job_title",
        label="Current Role",
        profile=sample_profile,
    )
    assert title_res == "Senior Python Engineer"

    # Contact defaults when not in profile model
    email_res = await mapper.resolve_field(
        field_id="email",
        label="Email Address",
        profile=sample_profile,
    )
    assert "@" in email_res

    location_res = await mapper.resolve_field(
        field_id="location",
        label="Location",
        profile=sample_profile,
    )
    assert "Israel" in location_res

    mock_llm_gateway.ask_question.assert_not_called()


# ===========================================================================
# 2. Deterministic ATS Screening Question Rules (Zero-Cost Bypass)
# ===========================================================================

@pytest.mark.asyncio
async def test_deterministic_work_authorization_and_sponsorship(
    mock_llm_gateway: MagicMock,
) -> None:
    """Verify work authorization and visa sponsorship questions resolve deterministically without LLM."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    # 1. Work authorization in Israel (text and boolean)
    auth_text = await mapper.resolve_field(
        field_id="work_auth",
        label="Are you legally authorized to work in Israel?",
        field_type="text",
    )
    assert auth_text == "Yes"

    auth_bool = await mapper.resolve_field(
        field_id="work_auth_cb",
        label="Legally authorized to work in Israel",
        field_type="checkbox",
    )
    assert auth_bool is True

    # 2. Visa sponsorship (text and boolean)
    sponsorship_text = await mapper.resolve_field(
        field_id="visa_sponsorship",
        label="Will you now or in the future require visa sponsorship?",
        field_type="text",
    )
    assert sponsorship_text == "No"

    sponsorship_bool = await mapper.resolve_field(
        field_id="visa_sponsorship_cb",
        label="Require visa sponsorship",
        field_type="checkbox",
    )
    assert sponsorship_bool is False

    mock_llm_gateway.ask_question.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_relocation_and_work_mode(
    mock_llm_gateway: MagicMock,
) -> None:
    """Verify relocation and remote/hybrid preferences resolve deterministically without LLM."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    # Relocation
    reloc_res = await mapper.resolve_field(
        field_id="relocation",
        label="Are you willing to relocate?",
        field_type="text",
    )
    assert "relocation" in reloc_res.lower() or "yes" in reloc_res.lower()

    # Work Mode
    mode_res = await mapper.resolve_field(
        field_id="work_mode",
        label="Preferred work arrangement (Remote / Hybrid / On-site)?",
        field_type="text",
    )
    assert "hybrid" in mode_res.lower() or "remote" in mode_res.lower()

    mock_llm_gateway.ask_question.assert_not_called()


@pytest.mark.asyncio
async def test_deterministic_notice_period_and_experience(
    mock_llm_gateway: MagicMock, sample_profile: CandidateProfile
) -> None:
    """Verify notice period, availability, and years of experience resolve deterministically."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    # Notice Period
    notice_text = await mapper.resolve_field(
        field_id="notice_period",
        label="What is your notice period / availability to start?",
        field_type="text",
    )
    assert "immediately" in notice_text.lower() or "weeks" in notice_text.lower()

    # Experience Years
    exp_num = await mapper.resolve_field(
        field_id="years_experience",
        label="Total years of software engineering experience",
        field_type="number",
        profile=sample_profile,
    )
    assert exp_num == 7

    # Salary Expectations
    salary_res = await mapper.resolve_field(
        field_id="salary_expectation",
        label="What is your desired salary / compensation expectation?",
        field_type="text",
    )
    assert "discussion" in salary_res.lower() or "compensation" in salary_res.lower()

    mock_llm_gateway.ask_question.assert_not_called()


# ===========================================================================
# 3. Dropdown / Select Option Matching
# ===========================================================================

@pytest.mark.asyncio
async def test_dropdown_option_selection(mock_llm_gateway: MagicMock) -> None:
    """Verify dropdown and radio options are selected accurately from choices."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    # Work Authorization dropdown
    auth_options = [
        "I require visa sponsorship",
        "I am legally authorized to work in Israel without sponsorship",
        "Other",
    ]
    matched_auth = await mapper.resolve_field(
        field_id="work_auth_select",
        label="Work Authorization Status in Israel",
        field_type="select",
        options=auth_options,
    )
    assert matched_auth == "I am legally authorized to work in Israel without sponsorship"

    # Visa sponsorship dropdown
    sponsor_options = [
        "Yes, I need visa sponsorship",
        "No, I do not require sponsorship",
    ]
    matched_sponsor = await mapper.resolve_field(
        field_id="sponsorship_select",
        label="Do you require visa sponsorship?",
        field_type="select",
        options=sponsor_options,
    )
    assert matched_sponsor == "No, I do not require sponsorship"

    # Experience Range dropdown
    exp_options = [
        "0 - 2 years",
        "3 - 5 years",
        "5 - 8 years",
        "8+ years",
    ]
    matched_exp = await mapper.resolve_field(
        field_id="exp_range",
        label="Years of Experience",
        field_type="select",
        options=exp_options,
    )
    assert matched_exp == "5 - 8 years"

    # Notice period dropdown
    notice_options = ["Immediate", "2 Weeks", "1 Month", "2 Months", "3+ Months"]
    matched_notice = await mapper.resolve_field(
        field_id="notice_select",
        label="Notice Period",
        field_type="select",
        options=notice_options,
    )
    assert matched_notice in ("Immediate", "2 Weeks")

    mock_llm_gateway.ask_question.assert_not_called()


# ===========================================================================
# 4. Context-Aware LLM Gateway Invocation & Caching for Custom Questions
# ===========================================================================

@pytest.mark.asyncio
async def test_ambiguous_question_invokes_llm_gateway(mock_llm_gateway: MagicMock) -> None:
    """Verify ambiguous/custom screening questions invoke the LLM gateway with CV context."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)
    mock_llm_gateway.ask_question.return_value = (
        "I have 6+ years building concurrent event-driven architectures with Python asyncio."
    )

    custom_q = "Describe your hands-on experience designing asynchronous systems with Python."
    cv_text = "Senior Python Engineer with deep asyncio, FastAPI, and distributed systems experience."

    answer = await mapper.resolve_field(
        field_id="custom_q1",
        label=custom_q,
        field_type="textarea",
        cv_text=cv_text,
    )

    assert "asyncio" in answer
    mock_llm_gateway.ask_question.assert_called_once()
    call_args = mock_llm_gateway.ask_question.call_args[1]
    assert call_args["question"] == custom_q
    assert "asyncio" in call_args["cv_context"]


@pytest.mark.asyncio
async def test_ambiguous_question_caching_with_live_gateway(
    live_memory_gateway: ResilientLLMGateway,
) -> None:
    """Verify ambiguous question responses are cached in LLMCache for zero-cost subsequent requests."""
    mapper = SemanticFormMapper(llm_gateway=live_memory_gateway)

    custom_q = "Why are you interested in joining our engineering team?"
    cv_text = "Passionate engineer focusing on scalable cloud platforms."

    # First call - uses mock LLM fallback and caches result in SQLite
    res1 = await mapper.resolve_field(
        field_id="why_us",
        label=custom_q,
        field_type="textarea",
        cv_text=cv_text,
    )
    assert len(res1) > 10

    # Verify cache has stored the entry
    cached = live_memory_gateway.cache.get_cached_answer(custom_q)
    assert cached == res1

    # Second call - must return instant cache hit
    res2 = await mapper.resolve_field(
        field_id="why_us",
        label=custom_q,
        field_type="textarea",
        cv_text=cv_text,
    )
    assert res2 == res1


# ===========================================================================
# 5. Full Form Schema Mapping (map_form_fields)
# ===========================================================================

@pytest.mark.asyncio
async def test_map_form_fields_mixed_schema(
    mock_llm_gateway: MagicMock, sample_profile_dict: dict
) -> None:
    """Verify map_form_fields resolves an entire ATS schema combining standard, rule-based, and LLM fields."""
    mock_llm_gateway.ask_question.return_value = "Led migration of monolith to microservices saving 40% latency."
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    schema = [
        {"id": "first_name", "label": "First Name", "type": "text"},
        {"id": "last_name", "label": "Last Name", "type": "text"},
        {"id": "email", "label": "Email Address", "type": "email"},
        {"id": "phone", "label": "Mobile Phone", "type": "tel"},
        {"id": "linkedin_url", "label": "LinkedIn Profile", "type": "url"},
        {"id": "work_auth", "label": "Authorized to work in Israel?", "type": "text"},
        {"id": "sponsorship", "label": "Need visa sponsorship?", "type": "checkbox"},
        {
            "id": "relocation_pref",
            "label": "Relocation Preferences",
            "type": "select",
            "options": ["Open to relocation / Remote", "Local only", "No"],
        },
        {"id": "biggest_achievement", "label": "Describe your greatest engineering achievement", "type": "textarea"},
    ]

    mapped = await mapper.map_form_fields(
        fields_schema=schema,
        profile=sample_profile_dict,
        cv_text="Lior Zvieli - Senior Architect",
    )

    assert mapped["first_name"] == "Lior"
    assert mapped["last_name"] == "Zvieli"
    assert mapped["email"] == "lior@example.com"
    assert mapped["phone"] == "+972-54-1234567"
    assert mapped["linkedin_url"] == "https://www.linkedin.com/in/liorzvieli"
    assert mapped["work_auth"] == "Yes"
    assert mapped["sponsorship"] is False
    assert "Open to relocation" in mapped["relocation_pref"]
    assert "microservices" in mapped["biggest_achievement"]

    # Only the custom achievement question should have called LLM!
    assert mock_llm_gateway.ask_question.call_count == 1


# ===========================================================================
# 6. Strategy Integration (ApiPostStrategy and BrowserPlaywrightStrategy)
# ===========================================================================

@pytest.mark.asyncio
async def test_api_post_strategy_integrates_mapper(
    mock_llm_gateway: MagicMock, sample_profile_dict: dict
) -> None:
    """Verify ApiPostStrategy leverages SemanticFormMapper for preview and apply resolution."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)
    strategy = ApiPostStrategy(form_mapper=mapper)

    job = Job(
        job_id="job-202",
        title="Senior Backend Engineer",
        company="Startup Nation Ltd",
        location="Tel Aviv, Israel",
        work_mode=WorkMode.HYBRID,
        source="comeet",
    )
    profile = CandidateProfile(
        skills=["Python", "FastAPI"],
        top_skills=["Python"],
        primary_stack=["Python"],
        seniority_level="Senior",
        target_roles=["Senior Backend Engineer"],
    )

    preview = await strategy.preview(job=job, profile=profile)
    assert preview.fields_to_submit["applicant_name"] != ""
    assert preview.fields_to_submit["work_authorization"] == "Yes"

    apply_res = await strategy.apply(job=job, profile=profile)
    assert apply_res["success"] is True
    assert apply_res["response"]["payload"]["work_authorization"] == "Yes"


@pytest.mark.asyncio
async def test_browser_playwright_strategy_integrates_mapper(
    mock_llm_gateway: MagicMock,
) -> None:
    """Verify BrowserPlaywrightStrategy leverages SemanticFormMapper for DOM field values."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)
    strategy = BrowserPlaywrightStrategy(form_mapper=mapper)

    job = Job(
        job_id="job-303",
        title="Principal Infrastructure Engineer",
        company="Global Enterprises",
        location="Herzliya, Israel",
        work_mode=WorkMode.ONSITE,
        source="workday",
    )
    profile = CandidateProfile(
        skills=["Kubernetes", "Terraform", "Go"],
        top_skills=["Kubernetes"],
        primary_stack=["Go"],
        seniority_level="Principal",
        target_roles=["Principal Infrastructure Engineer"],
    )

    preview = await strategy.preview(job=job, profile=profile)
    fields = preview.fields_to_submit
    assert "applicant_name" in fields
    assert fields["applicant_name"]["type"] == "text"
    assert fields["applicant_name"]["value"] != ""
    assert "@" in fields["applicant_email"]["value"]
