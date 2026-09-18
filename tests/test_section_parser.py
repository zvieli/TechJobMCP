"""Tests for bilingual Universal Section Parser and clean tech stack extraction."""

from job_mcp.core.section_parser import (
    JobSections,
    extract_clean_job_tech_stack,
    parse_job_sections,
)


class TestParseJobSectionsEnglishHtml:
    """Test parsing of English job descriptions with HTML markup."""

    def test_full_html_sections(self):
        html_content = """
        <div class="job-header">
            <h1>Staff Backend Engineer</h1>
            <p>Location: Tel Aviv, Israel</p>
        </div>
        <h2>About Us</h2>
        <p>Acme Corp is pioneering autonomous agentic AI platforms using Kubernetes, Python, and PyTorch.</p>
        <h3>About The Role</h3>
        <p>As a Staff Backend Engineer, you will:</p>
        <ul>
            <li>Architect and scale distributed backend microservices.</li>
            <li>Collaborate with cross-functional product and infrastructure teams.</li>
            <li>Lead architectural reviews and mentor junior engineers.</li>
        </ul>
        <h3>Qualifications</h3>
        <p>What we expect from you:</p>
        <ul>
            <li>7+ years of experience with Python, Go, or Java.</li>
            <li>Hands-on experience with PostgreSQL, Redis, and Kafka.</li>
            <li>Deep knowledge of Kubernetes and AWS cloud deployments.</li>
        </ul>
        <h3>What We Offer</h3>
        <p>Competitive equity package, comprehensive medical coverage, and flexible hybrid work policy.</p>
        """

        sections = parse_job_sections(html_content)

        assert isinstance(sections, JobSections)
        assert "Staff Backend Engineer" in sections.raw_other
        assert "pioneering autonomous agentic AI" in sections.company_overview
        assert "Architect and scale distributed backend microservices" in sections.responsibilities
        assert "7+ years of experience with Python" in sections.requirements
        assert "PostgreSQL, Redis, and Kafka" in sections.requirements
        assert "Competitive equity package" in sections.benefits

    def test_strong_tags_inline_headers(self):
        html_content = """
        <p><strong>Company Overview:</strong> CloudScale is an enterprise SaaS company.</p>
        <p><strong>Key Responsibilities:</strong></p>
        <p>Develop REST APIs in FastAPI.</p>
        <p><strong>Requirements:</strong> 5+ years with Python and Docker.</p>
        <p><strong>Perks & Benefits:</strong> Unlimited PTO and learning budget.</p>
        """
        sections = parse_job_sections(html_content)

        assert "CloudScale is an enterprise SaaS" in sections.company_overview
        assert "Develop REST APIs in FastAPI" in sections.responsibilities
        assert "5+ years with Python and Docker" in sections.requirements
        assert "Unlimited PTO" in sections.benefits


class TestParseJobSectionsHebrewText:
    """Test parsing of Hebrew job descriptions."""

    def test_full_hebrew_sections(self):
        hebrew_text = """
        חברת סטארטאפ מובילה בתחום הפינטק מגייסת!
        
        אודות החברה:
        אנחנו מפתחים פלטפורמת בינה מלאכותית מבוססת Python, Kubernetes ו-LLM לעולם הפיננסי.
        
        תיאור התפקיד:
        - פיתוח מערכות Real-Time עתירות ביצועים
        - הובלת תהליכי ארכיטקטורה ומיקרו-שירותים
        - עבודה צמודה עם צוותי ה-Data Science
        
        דרישות התפקיד:
        - לפחות 4 שנות ניסיון בפיתוח Python או Go
        - ניסיון מוכח בעבודה עם Docker ו-PostgreSQL
        - יתרון משמעותי: ניסיון עם AWS ו-Kafka
        
        מה אנחנו מציעים:
        - תנאי שכר מצוינים, קרן השתלמות מהיום הראשון, ומודל עבודה היברידי.
        """

        sections = parse_job_sections(hebrew_text)

        assert "פלטפורמת בינה מלאכותית" in sections.company_overview
        assert "פיתוח מערכות Real-Time" in sections.responsibilities
        assert "4 שנות ניסיון בפיתוח Python" in sections.requirements
        assert "Docker ו-PostgreSQL" in sections.requirements
        assert "קרן השתלמות מהיום הראשון" in sections.benefits

    def test_hebrew_header_variants(self):
        sample_1 = "על החברה:\nסטארטאפ בצמיחה.\nתחומי אחריות:\nפיתוח קוד.\nדרישות חובה:\nניסיון ב-Java."
        s1 = parse_job_sections(sample_1)
        assert "סטארטאפ בצמיחה" in s1.company_overview
        assert "פיתוח קוד" in s1.responsibilities
        assert "ניסיון ב-Java" in s1.requirements

        sample_2 = "מי אנחנו:\nצוות מובחר.\nבמסגרת התפקיד:\nתכנון פיצ'רים.\nמה אנחנו מחפשים:\nמומחה Python.\nתנאים והטבות:\nחדר כושר."
        s2 = parse_job_sections(sample_2)
        assert "צוות מובחר" in s2.company_overview
        assert "תכנון פיצ'רים" in s2.responsibilities
        assert "מומחה Python" in s2.requirements
        assert "חדר כושר" in s2.benefits


class TestHeaderVariations:
    """Test exhaustive header pattern matches for both languages."""

    def test_english_header_patterns(self):
        english_cases = [
            ("## Requirements", "requirements"),
            ("Qualifications:", "requirements"),
            ("What You Bring:", "requirements"),
            ("What you'll need:", "requirements"),
            ("What You'll Need", "requirements"),
            ("Who You Are:", "requirements"),
            ("Skills & Experience:", "requirements"),
            ("Must Have:", "requirements"),
            ("Basic Qualifications:", "requirements"),
            ("Preferred Qualifications:", "requirements"),
            ("Minimum Qualifications:", "requirements"),
            ("What We're Looking For:", "requirements"),
            ("### Responsibilities", "responsibilities"),
            ("What You'll Do:", "responsibilities"),
            ("The Role:", "responsibilities"),
            ("About The Role:", "responsibilities"),
            ("Your Impact:", "responsibilities"),
            ("Duties:", "responsibilities"),
            ("Scope of Work:", "responsibilities"),
            ("Key Responsibilities:", "responsibilities"),
            ("About Us", "company_overview"),
            ("About The Company:", "company_overview"),
            ("Who We Are:", "company_overview"),
            ("Company Overview:", "company_overview"),
            ("Our Mission:", "company_overview"),
            ("Life at Google:", "company_overview"),
            ("Working at Monday:", "company_overview"),
            ("Join Our Team:", "company_overview"),
            ("About Acme Corp:", "company_overview"),
            ("What We Offer:", "benefits"),
            ("Benefits:", "benefits"),
            ("Perks:", "benefits"),
            ("Why Join Us:", "benefits"),
            ("Compensation & Benefits:", "benefits"),
        ]

        for header_line, expected_section in english_cases:
            text = f"{header_line}\nSample text content for section."
            sections = parse_job_sections(text)
            val = getattr(sections, expected_section)
            assert "Sample text content for section." in val, f"Failed for header: {header_line}"

    def test_hebrew_header_patterns(self):
        hebrew_cases = [
            ("דרישות:", "requirements"),
            ("דרישות התפקיד:", "requirements"),
            ("דרישות סף:", "requirements"),
            ("דרישות חובה:", "requirements"),
            ("כישורים:", "requirements"),
            ("מה אנחנו מחפשים:", "requirements"),
            ("מה תביא איתך:", "requirements"),
            ("תחומי אחריות:", "responsibilities"),
            ("תיאור התפקיד:", "responsibilities"),
            ("במסגרת התפקיד:", "responsibilities"),
            ("מהות התפקיד:", "responsibilities"),
            ("הגדרת התפקיד:", "responsibilities"),
            ("אודות החברה:", "company_overview"),
            ("על החברה:", "company_overview"),
            ("מי אנחנו:", "company_overview"),
            ("קצת עלינו:", "company_overview"),
            ("על הארגון:", "company_overview"),
            ("מה אנחנו מציעים:", "benefits"),
            ("תנאים והטבות:", "benefits"),
            ("למה לעבוד אצלנו:", "benefits"),
        ]

        for header_line, expected_section in hebrew_cases:
            text = f"{header_line}\nתוכן דוגמה עבור החלק."
            sections = parse_job_sections(text)
            val = getattr(sections, expected_section)
            assert "תוכן דוגמה עבור החלק." in val, f"Failed for hebrew header: {header_line}"


class TestCleanJobTechStackExtraction:
    """Test clean tech stack extraction and boilerplate isolation."""

    def test_boilerplate_isolation_non_tech_role(self):
        """Social Media Manager whose company overview mentions heavy tech buzzwords."""
        html_content = """
        <h2>About Acme AI</h2>
        <p>Acme is building the future of autonomous agentic AI, leveraging Python, Kubernetes, PyTorch, and Docker.</p>
        <h3>What You'll Do</h3>
        <p>Manage company Twitter and LinkedIn accounts, increase social engagement, and draft campaign posts.</p>
        <h3>Requirements</h3>
        <p>3+ years of experience in B2B social media marketing and brand growth.</p>
        <h3>Benefits</h3>
        <p>Health insurance, generous vacation, and commuter stipend.</p>
        """
        sections = parse_job_sections(html_content)
        tech_stack = extract_clean_job_tech_stack(
            title="Social Media Manager",
            sections=sections,
            department="Marketing",
        )

        assert tech_stack == [], f"Expected empty tech stack, but got: {tech_stack}"

    def test_technical_role_extraction(self):
        """AI Engineer with technical skills in requirements and responsibilities."""
        html_content = """
        <h2>About Us</h2>
        <p>We are a high-growth tech startup.</p>
        <h3>What You'll Do</h3>
        <p>Build and evaluate LLM pipelines and fine-tune foundation models.</p>
        <h3>Requirements</h3>
        <p>Strong programming skills in Python. Experience with PyTorch, LangChain, and Docker.</p>
        """
        sections = parse_job_sections(html_content)
        tech_stack = extract_clean_job_tech_stack(
            title="Senior AI Engineer",
            sections=sections,
            department="R&D",
        )

        tech_lower = [t.lower() for t in tech_stack]
        assert "python" in tech_lower
        assert "pytorch" in tech_lower
        assert "docker" in tech_lower
        assert "llm" in tech_lower

    def test_unstructured_fallback_technical_role(self):
        """Unstructured text without section headers for a technical role."""
        raw_text = "Looking for a Python Developer with Docker and AWS experience to join our team."
        sections = parse_job_sections(raw_text)

        assert sections.requirements == ""
        assert sections.responsibilities == ""
        assert raw_text in sections.raw_other

        tech_stack = extract_clean_job_tech_stack(
            title="Python Developer",
            sections=sections,
            department="Engineering",
            fallback_text=raw_text,
        )

        tech_lower = [t.lower() for t in tech_stack]
        assert "python" in tech_lower
        assert "docker" in tech_lower
        assert "aws" in tech_lower

    def test_unstructured_fallback_non_tech_role_guarantee(self):
        """Non-tech role with unstructured text mentioning tech terms must return empty list."""
        raw_text = "Seeking Recruiter to hire engineers skilled in Python, React, and AWS."
        sections = parse_job_sections(raw_text)

        tech_stack = extract_clean_job_tech_stack(
            title="Talent Acquisition Specialist",
            sections=sections,
            department="Human Resources",
            fallback_text=raw_text,
        )

        assert tech_stack == [], f"Expected empty tech stack for non-tech role, got: {tech_stack}"

    def test_department_non_tech_override(self):
        """Role with generic title but non-tech department gets empty stack in fallback."""
        raw_text = "General coordinator assisting teams with Python automation tools."
        sections = parse_job_sections(raw_text)

        tech_stack = extract_clean_job_tech_stack(
            title="Coordinator",
            sections=sections,
            department="People Operations",
            fallback_text=raw_text,
        )

        assert tech_stack == []

    def test_empty_or_whitespace_input(self):
        """Gracefully handle empty or whitespace inputs."""
        sections = parse_job_sections("")
        assert sections == JobSections()

        tech_stack = extract_clean_job_tech_stack(
            title="",
            sections=sections,
            department=None,
            fallback_text=None,
        )
        assert tech_stack == []
