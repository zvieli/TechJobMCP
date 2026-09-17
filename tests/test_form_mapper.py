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
        "first_name": "Jane",
        "last_name": "Doe",
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "+972-54-1234567",
        "linkedin": "https://www.linkedin.com/in/janedoe",
        "github": "https://github.com/janedoe",
        "portfolio": "https://janedoe.dev",
        "location": "Tel Aviv, Israel",
        "current_title": "Lead Software Architect",
        "current_company": "CloudTech Systems",
        "cv_path": "/path/to/cv.pdf",
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
        ("first_name", "First Name", "text", "Jane"),
        ("fname", "Given Name", "text", "Jane"),
        ("last_name", "Last Name", "text", "Doe"),
        ("lname", "Surname / Family Name", "text", "Doe"),
        ("full_name", "Full Name", "text", "Jane Doe"),
        ("applicant_name", "Candidate Name", "text", "Jane Doe"),
        ("email", "Email Address", "email", "jane@example.com"),
        ("applicant_email", "Primary E-mail", "email", "jane@example.com"),
        ("phone", "Phone Number", "tel", "+972-54-1234567"),
        ("mobile_number", "Mobile / Cell", "tel", "+972-54-1234567"),
        ("linkedin_url", "LinkedIn Profile URL", "url", "https://www.linkedin.com/in/janedoe"),
        ("github_url", "GitHub Profile", "url", "https://github.com/janedoe"),
        ("portfolio_url", "Personal Website / Portfolio", "url", "https://janedoe.dev"),
        ("location", "Current City / Location", "text", "Tel Aviv, Israel"),
        ("current_title", "Current Job Title / Role", "text", "Lead Software Architect"),
        ("current_company", "Current Employer / Company", "text", "CloudTech Systems"),
        ("resume_file", "Upload Resume / CV", "file", "/path/to/cv.pdf"),
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
        field_id="custom_q2",
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
        field_id="custom_q2",
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
        cv_text="Jane Doe - Senior Architect",
    )

    assert mapped["first_name"] == "Jane"
    assert mapped["last_name"] == "Doe"
    assert mapped["email"] == "jane@example.com"
    assert mapped["phone"] == "+972-54-1234567"
    assert mapped["linkedin_url"] == "https://www.linkedin.com/in/janedoe"
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


# ===========================================================================
# 7. Hebrew Form Field Mapping (Zero-Cost Regex Heuristics)
# ===========================================================================

@pytest.mark.asyncio
async def test_hebrew_standard_form_fields(
    mock_llm_gateway: MagicMock, sample_profile_dict: dict
) -> None:
    """Verify Hebrew form labels match standard candidate fields with zero LLM calls."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    test_cases = [
        # Full name: שם מלא, שם המועמד, שם
        ("field_1", "שם מלא", "text", sample_profile_dict["full_name"]),
        ("field_2", "שם המועמד", "text", sample_profile_dict["full_name"]),
        ("field_3", "שם", "text", sample_profile_dict["full_name"]),
        # First name: שם פרטי
        ("field_4", "שם פרטי", "text", sample_profile_dict["first_name"]),
        # Last name: שם משפחה
        ("field_5", "שם משפחה", "text", sample_profile_dict["last_name"]),
        # Email: דוא"ל, אימייל, מייל, כתובת מייל
        ("field_6", 'דוא"ל', "email", sample_profile_dict["email"]),
        ("field_7", "אימייל", "email", sample_profile_dict["email"]),
        ("field_8", "מייל", "email", sample_profile_dict["email"]),
        ("field_9", "כתובת מייל", "email", sample_profile_dict["email"]),
        # Phone: טלפון, נייד, טלפון נייד, מספר טלפון, סלולרי
        ("field_10", "טלפון", "tel", sample_profile_dict["phone"]),
        ("field_11", "נייד", "tel", sample_profile_dict["phone"]),
        ("field_12", "טלפון נייד", "tel", sample_profile_dict["phone"]),
        ("field_13", "מספר טלפון", "tel", sample_profile_dict["phone"]),
        ("field_14", "סלולרי", "tel", sample_profile_dict["phone"]),
        # CV: קורות חיים, קובץ קורות חיים, צרף קו"ח, צרף קובץ, קו"ח, קו״ח
        ("field_15", "קורות חיים", "file", sample_profile_dict["cv_path"]),
        ("field_16", "קובץ קורות חיים", "file", sample_profile_dict["cv_path"]),
        ("field_17", 'צרף קו"ח', "file", sample_profile_dict["cv_path"]),
        ("field_18", "צרף קובץ", "file", sample_profile_dict["cv_path"]),
        ("field_19", 'קו"ח', "file", sample_profile_dict["cv_path"]),
        ("field_20", "קו״ח", "file", sample_profile_dict["cv_path"]),
        # Location: עיר מגורים, מגורים, כתובת, עיר, יישוב
        ("field_21", "עיר מגורים", "text", sample_profile_dict["location"]),
        ("field_22", "מגורים", "text", sample_profile_dict["location"]),
        ("field_23", "כתובת", "text", sample_profile_dict["location"]),
        ("field_24", "עיר", "text", sample_profile_dict["location"]),
        ("field_25", "יישוב", "text", sample_profile_dict["location"]),
    ]

    for field_id, label, ftype, expected_val in test_cases:
        res = await mapper.resolve_field(
            field_id=field_id,
            label=label,
            field_type=ftype,
            profile=sample_profile_dict,
        )
        assert res == expected_val, f"Failed resolving Hebrew label: {label} (field_id={field_id})"

    mock_llm_gateway.ask_question.assert_not_called()


@pytest.mark.asyncio
async def test_hebrew_work_authorization(mock_llm_gateway: MagicMock) -> None:
    """Verify Hebrew work authorization questions resolve deterministically without LLM."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    for phrase in ("אזרחות ישראלית", "אישור עבודה", "מורשה לעבוד בישראל"):
        text_val = await mapper.resolve_field(
            field_id="work_auth_heb",
            label=f"האם יש לך {phrase}?",
            field_type="text",
        )
        assert text_val == "Yes", f"Failed resolving Hebrew work auth text: {phrase}"

        bool_val = await mapper.resolve_field(
            field_id="work_auth_heb_cb",
            label=phrase,
            field_type="checkbox",
        )
        assert bool_val is True, f"Failed resolving Hebrew work auth checkbox: {phrase}"

    mock_llm_gateway.ask_question.assert_not_called()


@pytest.mark.asyncio
async def test_hebrew_full_form_mapping(
    mock_llm_gateway: MagicMock, sample_profile_dict: dict
) -> None:
    """Verify map_form_fields handles an entire Hebrew ATS application form."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    schema = [
        {"id": "name_heb", "label": "שם מלא", "type": "text"},
        {"id": "email_heb", "label": "כתובת מייל", "type": "email"},
        {"id": "phone_heb", "label": "טלפון נייד", "type": "tel"},
        {"id": "cv_heb", "label": "קובץ קורות חיים", "type": "file"},
        {"id": "city_heb", "label": "עיר מגורים", "type": "text"},
        {"id": "auth_heb", "label": "מורשה לעבוד בישראל", "type": "checkbox"},
    ]

    mapped = await mapper.map_form_fields(
        fields_schema=schema,
        profile=sample_profile_dict,
    )

    assert mapped["name_heb"] == "Jane Doe"
    assert mapped["email_heb"] == "jane@example.com"
    assert mapped["phone_heb"] == "+972-54-1234567"
    assert mapped["cv_heb"] == "/path/to/cv.pdf"
    assert mapped["city_heb"] == "Tel Aviv, Israel"
    assert mapped["auth_heb"] is True

    mock_llm_gateway.ask_question.assert_not_called()


@pytest.mark.asyncio
async def test_consent_and_terms_checkbox_resolves_true(mock_llm_gateway: MagicMock) -> None:
    """Verify terms, consent, privacy, and agreement checkboxes resolve to True deterministically."""
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    test_cases = [
        ("agree_terms", "I agree to the terms and conditions", "checkbox", True),
        ("privacy_policy", "I accept the Privacy Policy", "checkbox", True),
        ("consent", "Candidate Consent", "bool", True),
        ("gdpr_consent", "GDPR agreement", "checkbox", True),
        ("terms", "Terms of Service", "boolean", True),
        ("marketing_opt_in", "Agree to receive marketing updates", "checkbox", True),
        ("takanoon", "אישור תקנון ותנאי שימוש", "checkbox", True),
        ("agree_hebrew", "אני מסכים לתנאים", "checkbox", True),
    ]

    for field_id, label, ftype, expected in test_cases:
        res = await mapper.resolve_field(
            field_id=field_id,
            label=label,
            field_type=ftype,
        )
        assert res is expected, f"Failed resolving {field_id} ({label})"

    mock_llm_gateway.ask_question.assert_not_called()


@pytest.mark.asyncio
async def test_cover_letter_and_personal_note_generation(
    mock_llm_gateway: MagicMock, sample_profile: CandidateProfile
) -> None:
    """Verify cover letter, personal note, and comments fields invoke generate_personal_note."""
    mock_llm_gateway.generate_personal_note = AsyncMock(return_value="Tailored personal note for the role.")
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    job = Job(
        job_id="job_123",
        title="Staff ML Engineer",
        company="CyberCorp",
        description="Looking for Python and ML specialists.",
        apply_url="https://cybercorp.com/apply",
        source="career_page",
    )

    test_fields = [
        ("cover_letter", "Cover Letter", "textarea"),
        ("personal_note", "Personal Note to Hiring Manager", "text"),
        ("comments", "Additional Comments / Info", "textarea"),
        ("why_join", "Why do you want to join us?", "textarea"),
        ("hebrew_note", "מכתב מקדים והערות", "textarea"),
    ]

    for field_id, label, ftype in test_fields:
        res = await mapper.resolve_field(
            field_id=field_id,
            label=label,
            field_type=ftype,
            profile=sample_profile,
            job=job,
        )
        assert res == "Tailored personal note for the role."

    assert mock_llm_gateway.generate_personal_note.call_count == len(test_fields)
    call_kwargs = mock_llm_gateway.generate_personal_note.call_args.kwargs
    assert call_kwargs["job_title"] == "Staff ML Engineer"
    assert call_kwargs["company"] == "CyberCorp"
    assert call_kwargs["job_description"] == "Looking for Python and ML specialists."
    assert "Candidate Seniority" in call_kwargs["cv_context"]
    assert "Python" in call_kwargs["cv_context"]


@pytest.mark.asyncio
async def test_candidate_profile_direct_attributes(mock_llm_gateway: MagicMock) -> None:
    """Verify CandidateProfile direct attributes (full_name, email, phone) are resolved."""
    profile = CandidateProfile(
        full_name="Gal Gadot",
        first_name="Gal",
        last_name="Gadot",
        email="gal@gadot.com",
        phone="+972-50-7654321",
        skills=["Acting", "Python"],
    )
    mapper = SemanticFormMapper(llm_gateway=mock_llm_gateway)

    name_res = await mapper.resolve_field(
        field_id="applicant_name",
        label="Full Name",
        field_type="text",
        profile=profile,
    )
    assert name_res == "Gal Gadot"

    email_res = await mapper.resolve_field(
        field_id="email",
        label="Email Address",
        field_type="email",
        profile=profile,
    )
    assert email_res == "gal@gadot.com"

    phone_res = await mapper.resolve_field(
        field_id="phone_number",
        label="Mobile Phone",
        field_type="tel",
        profile=profile,
    )
    assert phone_res == "+972-50-7654321"

    mock_llm_gateway.ask_question.assert_not_called()


