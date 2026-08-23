"""Semantic Form Field Mapper and Questionnaire Solver for ATS applications."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

from job_mcp.core.llm.gateway import ResilientLLMGateway
from job_mcp.models.schemas import CandidateProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zero-Cost Regex Heuristic Patterns for Standard Candidate Fields
# ---------------------------------------------------------------------------

FIRST_NAME_REGEX = re.compile(
    r"^(first[_\s\-]*name|fname|given[_\s\-]*name|first)$|\b(first[_\s\-]*name|given[_\s\-]*name)\b",
    re.IGNORECASE,
)
LAST_NAME_REGEX = re.compile(
    r"^(last[_\s\-]*name|lname|surname|family[_\s\-]*name|last)$|\b(last[_\s\-]*name|family[_\s\-]*name|surname)\b",
    re.IGNORECASE,
)
FULL_NAME_REGEX = re.compile(
    r"^(full[_\s\-]*name|fullname|candidate[_\s\-]*name|applicant[_\s\-]*name|your[_\s\-]*name|name)$|\b(full[_\s\-]*name|candidate[_\s\-]*name|applicant[_\s\-]*name)\b",
    re.IGNORECASE,
)
EMAIL_REGEX = re.compile(
    r"\b(e[_\s\-]*mail|email[_\s\-]*address|applicant[_\s\-]*email|primary[_\s\-]*email|mail)\b",
    re.IGNORECASE,
)
PHONE_REGEX = re.compile(
    r"\b(phone|mobile|cell|telephone|tel|phone[_\s\-]*number|mobile[_\s\-]*number|contact[_\s\-]*number)\b",
    re.IGNORECASE,
)
LINKEDIN_REGEX = re.compile(
    r"\b(linkedin|linked[_\s\-]*in|linkedin[_\s\-]*url|linkedin[_\s\-]*profile)\b",
    re.IGNORECASE,
)
GITHUB_REGEX = re.compile(
    r"\b(github|git[_\s\-]*hub|github[_\s\-]*url|github[_\s\-]*profile)\b",
    re.IGNORECASE,
)
PORTFOLIO_REGEX = re.compile(
    r"^(portfolio|website|personal[_\s\-]*website|personal[_\s\-]*url|personal[_\s\-]*site|homepage|blog)$|\b(portfolio|personal[_\s\-]*website|personal[_\s\-]*url|personal[_\s\-]*site|homepage|my[_\s\-]*website|blog)\b",
    re.IGNORECASE,
)
LOCATION_REGEX = re.compile(
    r"^(location|city|country|address|residence|current[_\s\-]*location|state|zip|postal[_\s\-]*code)$|\b(current[_\s\-]*location|current[_\s\-]*city|residence[_\s\-]*city|home[_\s\-]*address|postal[_\s\-]*code)\b|\b(location|city|country|address)\b",
    re.IGNORECASE,
)
TITLE_REGEX = re.compile(
    r"\b(current[_\s\-]*title|current[_\s\-]*role|current[_\s\-]*position|job[_\s\-]*title|headline|occupation|role|title|position)\b",
    re.IGNORECASE,
)
COMPANY_REGEX = re.compile(
    r"\b(current[_\s\-]*company|current[_\s\-]*employer|company|employer|organization|workplace)\b",
    re.IGNORECASE,
)
CV_REGEX = re.compile(
    r"\b(resume|cv|resume[_\s\-]*file|cv[_\s\-]*file|resume[_\s\-]*path|cv[_\s\-]*path|upload[_\s\-]*resume|upload[_\s\-]*cv|attachment)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Deterministic Rule Regexes for Standard ATS Screening Questions
# ---------------------------------------------------------------------------

SPONSORSHIP_REGEX = re.compile(
    r"\b(sponsor(ship)?|require\s+(visa|sponsorship)|need\s+(visa|sponsorship)|visa\s+status)\b",
    re.IGNORECASE,
)
WORK_AUTH_REGEX = re.compile(
    r"\b(authoriz(ed|ation)|legal(ly)?\s+work|eligible\s+to\s+work|right\s+to\s+work|work\s+permit|israel(i)?\s+citizen(ship)?|work\s+in\s+israel|permitted\s+to\s+work)\b",
    re.IGNORECASE,
)
RELOCATION_REGEX = re.compile(
    r"\b(relocat(e|ion)|willing\s+to\s+relocate|open\s+to\s+relocate)\b",
    re.IGNORECASE,
)
WORK_MODE_REGEX = re.compile(
    r"\b(remote|hybrid|on[_\s\-]*site|work[_\s\-]*arrangement|preferred[_\s\-]*work[_\s\-]*mode|work[_\s\-]*mode|work[_\s\-]*from[_\s\-]*home)\b",
    re.IGNORECASE,
)
NOTICE_PERIOD_REGEX = re.compile(
    r"\b(notice[_\s\-]*period|how\s+soon\s+can\s+you\s+start|availability|available\s+to\s+start|start\s+date|earliest\s+start)\b",
    re.IGNORECASE,
)
EXPERIENCE_YEARS_REGEX = re.compile(
    r"\b(years?[_\s\-]*(of)?[_\s\-]*.*exp(erience)?|total[_\s\-]*exp(erience)?|how\s+many\s+years)\b",
    re.IGNORECASE,
)
SALARY_REGEX = re.compile(
    r"\b(salary|compensation|expected\s+salary|desired\s+salary|salary\s+expectation|rate)\b",
    re.IGNORECASE,
)
EDUCATION_REGEX = re.compile(
    r"\b(degree|education|highest\s+degree|bachelor|academic|university)\b",
    re.IGNORECASE,
)


class SemanticFormMapper:
    """Intelligent form field mapper combining zero-cost regex heuristics with LLM questionnaire solving."""

    def __init__(self, llm_gateway: Optional[ResilientLLMGateway] = None) -> None:
        """Initialize SemanticFormMapper.

        Args:
            llm_gateway: Optional ResilientLLMGateway instance. If omitted, a default instance is created.
        """
        self.llm_gateway = llm_gateway or ResilientLLMGateway()

    def _extract_profile_dict(
        self, profile: Optional[dict[str, Any] | CandidateProfile]
    ) -> dict[str, Any]:
        """Normalize CandidateProfile or dict into a standard key-value dictionary."""
        if profile is None:
            return {}
        if isinstance(profile, dict):
            return dict(profile)
        if isinstance(profile, CandidateProfile):
            d: dict[str, Any] = profile.model_dump()
            return d
        if hasattr(profile, "__dict__"):
            return dict(profile.__dict__)
        return {}

    def _get_candidate_attribute(
        self, profile_data: dict[str, Any], attr: str, default: Any = ""
    ) -> Any:
        """Retrieve candidate attribute with fallback aliases."""
        if not profile_data:
            return default

        if attr in profile_data and profile_data[attr]:
            return profile_data[attr]

        # Alias lookups
        if attr == "first_name":
            if "name" in profile_data and profile_data["name"]:
                return str(profile_data["name"]).split()[0]
            if "full_name" in profile_data and profile_data["full_name"]:
                return str(profile_data["full_name"]).split()[0]
        elif attr == "last_name":
            if "name" in profile_data and profile_data["name"]:
                parts = str(profile_data["name"]).split()
                return parts[-1] if len(parts) > 1 else ""
            if "full_name" in profile_data and profile_data["full_name"]:
                parts = str(profile_data["full_name"]).split()
                return parts[-1] if len(parts) > 1 else ""
        elif attr == "full_name":
            first = profile_data.get("first_name", "")
            last = profile_data.get("last_name", "")
            if first or last:
                return f"{first} {last}".strip()
            if "name" in profile_data and profile_data["name"]:
                return profile_data["name"]
        elif attr == "current_title":
            target_roles = profile_data.get("target_roles", [])
            if target_roles and isinstance(target_roles, list) and len(target_roles) > 0:
                return target_roles[0]
            seniority = profile_data.get("seniority_level", "")
            if seniority:
                return f"{seniority} Software Engineer"
        elif attr == "cv_path":
            for k in ("cv_path", "resume_path", "cv_file", "resume_file"):
                if k in profile_data and profile_data[k]:
                    return profile_data[k]

        return profile_data.get(attr, default)

    def _match_dropdown_option(
        self, options: Sequence[str], target_val: Any
    ) -> Optional[str]:
        """Match target value against available dropdown / radio options using heuristic rules."""
        if not options:
            return None

        # 1. Boolean target matching
        if isinstance(target_val, bool):
            if target_val is True:
                for opt in options:
                    clean = opt.strip().lower()
                    if clean in ("yes", "true", "y", "1") or clean.startswith("yes"):
                        return opt
                    if "authorized" in clean or "citizen" in clean or "immediate" in clean:
                        return opt
                return options[0]
            else:
                for opt in options:
                    clean = opt.strip().lower()
                    if clean in ("no", "false", "n", "0") or clean.startswith("no"):
                        return opt
                    if "not required" in clean or "none" in clean or "no sponsor" in clean:
                        return opt
                return options[-1]

        target_str = str(target_val).strip().lower()
        if not target_str:
            return options[0]

        # 2. Exact case-insensitive match
        for opt in options:
            if opt.strip().lower() == target_str:
                return opt

        # 3. Substring match
        for opt in options:
            opt_lower = opt.strip().lower()
            if target_str in opt_lower or opt_lower in target_str:
                return opt

        # 4. Keyword heuristics for common ATS answers
        if target_str in ("yes", "true") or "yes" in target_str:
            for opt in options:
                opt_lower = opt.strip().lower()
                if opt_lower.startswith("yes") or opt_lower in ("true", "y", "1"):
                    return opt
        elif target_str in ("no", "false") or "no" in target_str:
            for opt in options:
                opt_lower = opt.strip().lower()
                if opt_lower.startswith("no") or opt_lower in ("false", "n", "0"):
                    return opt

        # 5. Experience / Number matching in ranges (e.g., target 7 matching '5-7 years' or '7+ years')
        digits = re.findall(r"\d+", target_str)
        if digits:
            num = int(digits[0])
            for opt in options:
                opt_digits = [int(d) for d in re.findall(r"\d+", opt)]
                if opt_digits:
                    if len(opt_digits) == 1 and num >= opt_digits[0]:
                        return opt
                    elif len(opt_digits) >= 2 and opt_digits[0] <= num <= opt_digits[1]:
                        return opt

        return None

    def _resolve_standard_field_heuristic(
        self,
        identifier: str,
        profile_data: dict[str, Any],
        field_type: str,
    ) -> Optional[Any]:
        """Resolve standard candidate personal and contact fields using zero-cost regex heuristics.

        Returns resolved value if matched, or None if not a standard contact field.
        """
        # Exclude questions that are actually work mode / relocation / screening questions
        if WORK_MODE_REGEX.search(identifier) or RELOCATION_REGEX.search(identifier):
            return None

        # First Name
        if FIRST_NAME_REGEX.search(identifier):
            val = self._get_candidate_attribute(profile_data, "first_name", "Candidate")
            return val

        # Last Name
        if LAST_NAME_REGEX.search(identifier):
            val = self._get_candidate_attribute(profile_data, "last_name", "Applicant")
            return val

        # Full Name
        if FULL_NAME_REGEX.search(identifier):
            val = self._get_candidate_attribute(profile_data, "full_name", "Candidate Applicant")
            return val

        # Email
        if EMAIL_REGEX.search(identifier):
            val = self._get_candidate_attribute(profile_data, "email", "candidate@example.com")
            return val

        # Phone
        if PHONE_REGEX.search(identifier):
            val = self._get_candidate_attribute(profile_data, "phone", "+972-50-0000000")
            return val

        # LinkedIn
        if LINKEDIN_REGEX.search(identifier):
            val = self._get_candidate_attribute(
                profile_data, "linkedin", "https://www.linkedin.com/in/candidate"
            )
            return val

        # GitHub
        if GITHUB_REGEX.search(identifier):
            val = self._get_candidate_attribute(
                profile_data, "github", "https://github.com/candidate"
            )
            return val

        # Portfolio / Website
        if PORTFOLIO_REGEX.search(identifier):
            val = self._get_candidate_attribute(
                profile_data, "portfolio", "https://candidate.dev"
            )
            return val

        # Location / Address / City / Country
        if LOCATION_REGEX.search(identifier):
            val = self._get_candidate_attribute(profile_data, "location", "Tel Aviv, Israel")
            return val

        # Current Title / Role
        if TITLE_REGEX.search(identifier):
            val = self._get_candidate_attribute(
                profile_data, "current_title", "Senior Software Engineer"
            )
            return val

        # Current Company / Employer
        if COMPANY_REGEX.search(identifier):
            val = self._get_candidate_attribute(
                profile_data, "current_company", "Tech Innovations"
            )
            return val

        # Resume / CV path
        if CV_REGEX.search(identifier):
            val = self._get_candidate_attribute(profile_data, "cv_path", "/path/to/cv.pdf")
            return val

        return None

    def _resolve_screening_question_heuristic(
        self,
        identifier: str,
        field_type: str,
        options: Optional[Sequence[str]] = None,
        profile_data: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Resolve standard ATS screening questions using deterministic heuristics without LLM.

        Returns resolved value if deterministically answered, or None if ambiguous/custom.
        """
        pdata = profile_data or {}
        is_bool_type = field_type.lower() in ("checkbox", "bool", "boolean")

        # 1. Visa Sponsorship Question (e.g. "Do you require sponsorship?")
        if SPONSORSHIP_REGEX.search(identifier):
            if is_bool_type:
                return False
            if options:
                matched = self._match_dropdown_option(options, "no")
                if not matched:
                    matched = self._match_dropdown_option(options, "not required")
                return matched if matched is not None else options[-1]
            return "No"

        # 2. Work Authorization Question (e.g. "Are you authorized to work in Israel?")
        if WORK_AUTH_REGEX.search(identifier):
            if is_bool_type:
                return True
            if options:
                matched = self._match_dropdown_option(options, "authorized")
                if not matched:
                    matched = self._match_dropdown_option(options, "citizen")
                if not matched:
                    matched = self._match_dropdown_option(options, "yes")
                return matched if matched is not None else options[0]
            return "Yes"

        # 3. Relocation Question (e.g. "Are you open to relocation?")
        if RELOCATION_REGEX.search(identifier):
            if is_bool_type:
                return True
            if options:
                matched = self._match_dropdown_option(options, "relocat")
                if not matched:
                    matched = self._match_dropdown_option(options, "open")
                if not matched:
                    matched = self._match_dropdown_option(options, "yes")
                return matched if matched is not None else options[0]
            return "Yes, open to relocation or remote work arrangements."

        # 4. Remote / Hybrid Work Mode Question
        if WORK_MODE_REGEX.search(identifier):
            if is_bool_type:
                return True
            if options:
                matched = self._match_dropdown_option(options, "hybrid")
                if not matched:
                    matched = self._match_dropdown_option(options, "remote")
                if not matched:
                    matched = self._match_dropdown_option(options, "yes")
                return matched if matched is not None else options[0]
            return "Comfortable with hybrid, remote, or on-site arrangements."

        # 5. Notice Period / Availability / Start Date
        if NOTICE_PERIOD_REGEX.search(identifier):
            if options:
                matched = self._match_dropdown_option(options, "Immediate")
                if not matched:
                    matched = self._match_dropdown_option(options, "2-4 weeks")
                return matched if matched is not None else options[0]
            if field_type.lower() in ("number", "integer"):
                return 0
            return "Available immediately (or 2-4 weeks notice)."

        # 6. Years of Experience
        if EXPERIENCE_YEARS_REGEX.search(identifier):
            seniority = str(pdata.get("seniority_level", "Senior")).lower()
            years = 7 if "senior" in seniority or "lead" in seniority else 4
            if field_type.lower() in ("number", "integer"):
                return years
            if options:
                matched = self._match_dropdown_option(options, f"{years}+")
                return matched if matched is not None else options[0]
            return f"{years}+ years of professional software engineering experience."

        # 7. Salary / Compensation Expectations
        if SALARY_REGEX.search(identifier):
            if field_type.lower() in ("number", "integer"):
                return 0
            if options:
                return options[0]
            return "Open to discussion based on total compensation and role responsibilities."

        # 8. Education / Degree
        if EDUCATION_REGEX.search(identifier):
            if options:
                matched = self._match_dropdown_option(options, "Bachelor")
                return matched if matched is not None else options[0]
            return "B.Sc. in Computer Science / Software Engineering."

        return None

    async def resolve_field(
        self,
        field_id: str,
        label: str,
        field_type: str = "text",
        options: Optional[list[str]] = None,
        profile: Optional[dict[str, Any] | CandidateProfile] = None,
        cv_text: Optional[str] = None,
    ) -> Any:
        """Resolve a single form field or screening question.

        Prioritizes:
        1. Zero-cost regex matching for standard profile/contact fields.
        2. Deterministic rule matching for standard ATS screening questions.
        3. Context-aware LLM generation with persistent caching for custom screening questionnaires.

        Args:
            field_id: ID or name attribute of the field (e.g. 'applicant_email', 'q_visa').
            label: Human-readable label or question text (e.g. 'Email Address', 'Why this role?').
            field_type: Field HTML/input type ('text', 'textarea', 'select', 'radio', 'checkbox', 'number', etc.).
            options: List of available options for dropdown/radio fields.
            profile: Candidate profile dict or CandidateProfile model.
            cv_text: Raw text or summary from CV / resume for LLM context grounding.

        Returns:
            Resolved field value (string, boolean, integer, or matching option).
        """
        profile_data = self._extract_profile_dict(profile)
        combined_text = f"{field_id} {label}".strip()

        # Step 1: Zero-cost regex matching for standard contact / profile fields
        std_val = self._resolve_standard_field_heuristic(
            combined_text, profile_data, field_type
        )
        if std_val is not None:
            if options and field_type.lower() in ("select", "radio"):
                matched_opt = self._match_dropdown_option(options, std_val)
                return matched_opt if matched_opt is not None else std_val
            return std_val

        # Step 2: Deterministic rule matching for standard ATS screening questions
        screen_val = self._resolve_screening_question_heuristic(
            combined_text, field_type, options, profile_data
        )
        if screen_val is not None:
            return screen_val

        # Step 3: Context-Aware LLM Gateway for open-ended or custom screening questions
        question_text = label.strip() if label.strip() else field_id.strip()

        # Construct CV context from profile or raw cv_text
        context = cv_text or ""
        if not context and profile_data:
            skills = profile_data.get("skills") or profile_data.get("top_skills") or []
            roles = profile_data.get("target_roles") or []
            seniority = profile_data.get("seniority_level", "")
            context = (
                f"Candidate Seniority: {seniority}\n"
                f"Target Roles: {', '.join(roles)}\n"
                f"Skills: {', '.join(skills)}"
            )

        if options and field_type.lower() in ("select", "radio"):
            # Format question for option selection
            opts_str = ", ".join([f"'{opt}'" for opt in options])
            prompt_q = (
                f"{question_text}\n\n"
                f"Available Options: [{opts_str}]\n"
                f"Choose the single most suitable option from the list above for this candidate."
            )
            raw_answer = await self.llm_gateway.ask_question(
                question=prompt_q, cv_context=context
            )
            matched = self._match_dropdown_option(options, raw_answer)
            return matched if matched is not None else options[0]

        # Freeform text / textarea question
        answer = await self.llm_gateway.ask_question(
            question=question_text, cv_context=context
        )
        return answer

    async def map_form_fields(
        self,
        fields_schema: list[dict[str, Any]],
        profile: Optional[dict[str, Any] | CandidateProfile] = None,
        cv_text: Optional[str] = None,
    ) -> dict[str, Any]:
        """Resolve and map an entire schema of ATS form fields.

        Args:
            fields_schema: List of field definitions. Each dict can include:
                - 'id' or 'field_id' or 'name': Identifier.
                - 'label' or 'title': Label or question string.
                - 'type' or 'field_type': Input type ('text', 'select', etc.).
                - 'options': Optional list of option strings.
                - 'required': Optional boolean.
            profile: Candidate profile dict or CandidateProfile model.
            cv_text: Optional raw CV text for context.

        Returns:
            dict[str, Any]: Mapping of {field_id: resolved_value}.
        """
        resolved: dict[str, Any] = {}

        for item in fields_schema:
            field_id = (
                item.get("id")
                or item.get("field_id")
                or item.get("name")
                or f"field_{len(resolved)}"
            )
            label = item.get("label") or item.get("title") or field_id
            field_type = item.get("type") or item.get("field_type") or "text"
            options = item.get("options")

            val = await self.resolve_field(
                field_id=str(field_id),
                label=str(label),
                field_type=str(field_type),
                options=options,
                profile=profile,
                cv_text=cv_text,
            )
            resolved[str(field_id)] = val

        return resolved
