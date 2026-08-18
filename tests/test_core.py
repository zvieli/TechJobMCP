"""Unit tests for hireme_mcp core modules."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import zipfile

from job_mcp.core.api_client import (
    JobCache,
    extract_cv_keywords,
    extract_dynamic_cv_skills,
    filter_jobs,
    resolve_cv_path,
)
from job_mcp.core.auth import (
    BASE_URL,
    DASHBOARD_PATH,
    LOGIN_PATH,
    SessionManager,
)
from job_mcp.core.browser import (
    SELECTORS,
    _extract_tech_from_text,
    _parse_work_mode,
    _resolve_selector,
    bookmark_job,
    delete_job,
    dynamic_registry,
    extract_jobs,
    preview_application,
    execute_application,
)
from job_mcp.models.schemas import Job, JobPreferences, WorkMode


class TestAuth(unittest.IsolatedAsyncioTestCase):
    """Tests for auth and session management."""

    def test_auth_constants(self):
        self.assertTrue(BASE_URL.startswith("https://hiremetech.com"))
        self.assertIn(DASHBOARD_PATH, ("/he-il/jobs-app", "/dashboard"))
        self.assertEqual(LOGIN_PATH, "/login")

    def test_session_manager_init_defaults(self):
        manager = SessionManager()
        self.assertIn(".hireme_mcp", str(manager.user_data_dir))
        self.assertTrue(manager.headless)

    def test_session_manager_custom_init(self):
        manager = SessionManager(user_data_dir="/tmp/test_profile", headless=False)
        self.assertEqual(str(manager.user_data_dir), "/tmp/test_profile")
        self.assertFalse(manager.headless)

    @patch("job_mcp.core.auth.async_playwright")
    async def test_session_manager_lifecycle(self, mock_async_playwright):
        mock_pw = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.is_closed = MagicMock(return_value=False)
        mock_context.pages = [mock_page]
        mock_pw.chromium.launch_persistent_context.return_value = mock_context

        # async_playwright() returns a PlaywrightContextManager whose .start() is an async coroutine returning mock_pw
        mock_cm = MagicMock()
        mock_cm.start = AsyncMock(return_value=mock_pw)
        mock_async_playwright.return_value = mock_cm

        manager = SessionManager(user_data_dir="/tmp/test_profile")
        self.assertFalse(manager.is_running)
        await manager.initialize()
        self.assertTrue(manager._initialized)
        self.assertTrue(manager.is_running)

        page = await manager.get_page()
        self.assertEqual(page, mock_page)

        # Test check_session_health - success
        mock_page.goto.return_value = MagicMock(status=200)
        mock_page.url = "https://hiremetech.com/dashboard"
        self.assertTrue(await manager.check_session_health())

        # Test check_session_health - redirected to login
        mock_page.url = "https://hiremetech.com/login"
        self.assertFalse(await manager.check_session_health())

        # Test check_session_health - 401 unauthorized
        mock_page.goto.return_value = MagicMock(status=401)
        mock_page.url = "https://hiremetech.com/dashboard"
        self.assertFalse(await manager.check_session_health())

        # Test inject_session_storage
        await manager.inject_session_storage({"token": "123", "state": {"a": 1}})
        mock_page.evaluate.assert_called_once()

        # Test shutdown
        await manager.shutdown()
        self.assertFalse(manager._initialized)
        self.assertFalse(manager.is_running)
        self.assertIsNone(manager.page)

    @patch("job_mcp.core.auth.async_playwright")
    async def test_session_manager_concurrency_lock(self, mock_async_playwright):
        """Test concurrent initialize() calls with asyncio.gather are safe and launch only once."""
        import asyncio
        mock_pw = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.is_closed = MagicMock(return_value=False)
        mock_context.pages = [mock_page]
        mock_pw.chromium.launch_persistent_context.return_value = mock_context

        mock_cm = MagicMock()
        mock_cm.start = AsyncMock(return_value=mock_pw)
        mock_async_playwright.return_value = mock_cm

        manager = SessionManager(user_data_dir="/tmp/test_profile_concurrency")
        self.assertFalse(manager.is_running)

        await asyncio.gather(
            manager.initialize(),
            manager.initialize(),
            manager.initialize(),
            manager.initialize(),
        )

        self.assertTrue(manager.is_running)
        self.assertTrue(manager._initialized)
        mock_cm.start.assert_awaited_once()
        mock_pw.chromium.launch_persistent_context.assert_awaited_once()

        await manager.shutdown()
        self.assertFalse(manager.is_running)
        self.assertFalse(manager._initialized)


class MockLocator:
    """Helper class to mock Playwright Locator behavior accurately."""

    def __init__(self, count_val=1, text="", attrs=None, children=None):
        self._count_val = count_val
        self._text = text
        self._attrs = attrs or {}
        self._children = children or []

    async def count(self):
        return self._count_val

    async def inner_text(self):
        return self._text

    async def get_attribute(self, name):
        return self._attrs.get(name)

    async def click(self):
        pass

    async def is_visible(self):
        return True

    @property
    def first(self):
        if self._children:
            return self._children[0]
        return self

    def nth(self, idx):
        if idx < len(self._children):
            return self._children[idx]
        return self

    def locator(self, sel):
        return self


class TestBrowser(unittest.IsolatedAsyncioTestCase):
    """Tests for browser selector registry and operations."""

    def setUp(self):
        dynamic_registry.clear()

    def tearDown(self):
        dynamic_registry.clear()

    def test_selectors_registry(self):
        required_keys = [
            "job_card",
            "job_title",
            "job_company",
            "bookmark_button",
            "delete_button",
            "apply_button",
        ]
        for key in required_keys:
            self.assertIn(key, SELECTORS)
            self.assertIn("primary", SELECTORS[key])
            self.assertIn("fallback", SELECTORS[key])

    def test_helper_extractors(self):
        self.assertEqual(_parse_work_mode("Remote position in NY"), WorkMode.REMOTE)
        self.assertEqual(_parse_work_mode("Hybrid role 2 days in office"), WorkMode.HYBRID)
        self.assertEqual(_parse_work_mode("On-site in London"), WorkMode.ONSITE)
        self.assertIsNone(_parse_work_mode("Full time position"))

        techs = _extract_tech_from_text("Looking for Python, FastAPI, and Docker expertise.")
        self.assertIn("Python", techs)
        self.assertIn("FastAPI", techs)
        self.assertIn("Docker", techs)

    async def test_resolve_selector_primary(self):
        mock_page = MagicMock()
        mock_loc = MockLocator(count_val=1)
        mock_page.locator.return_value = mock_loc

        res = await _resolve_selector(mock_page, "job_card")
        self.assertEqual(res, SELECTORS["job_card"]["primary"])

    async def test_extract_jobs_mock(self):
        mock_page = MagicMock()

        card = MockLocator(
            count_val=1,
            attrs={"data-job-id": "job-101", "class": "job-card"},
        )
        # Custom locator routing for card
        def card_locator_fn(sel):
            if "job-title" in sel or "title" in sel:
                return MockLocator(count_val=1, text="Staff Backend Engineer")
            elif "company" in sel:
                return MockLocator(count_val=1, text="Google DeepMind")
            elif "location" in sel:
                return MockLocator(count_val=1, text="Remote")
            elif "tech-badge" in sel or "badge" in sel:
                b1 = MockLocator(count_val=1, text="Python")
                b2 = MockLocator(count_val=1, text="FastMCP")
                return MockLocator(count_val=2, children=[b1, b2])
            elif "bookmark" in sel:
                return MockLocator(count_val=1, attrs={"class": "bookmark-btn active saved"})
            elif "description" in sel:
                return MockLocator(count_val=1, text="Exciting role building AI infrastructure")
            return MockLocator(count_val=0, text="")

        card.locator = card_locator_fn
        page_cards = MockLocator(count_val=1, children=[card])
        mock_page.locator = MagicMock(return_value=page_cards)

        jobs = await extract_jobs(mock_page)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, "job-101")
        self.assertEqual(jobs[0].title, "Staff Backend Engineer")
        self.assertEqual(jobs[0].company, "Google DeepMind")
        self.assertEqual(jobs[0].work_mode, WorkMode.REMOTE)
        self.assertTrue(jobs[0].is_bookmarked)

    async def test_bookmark_and_delete_job(self):
        mock_page = MagicMock()
        btn = MockLocator(count_val=1)
        card = MockLocator(count_val=1, children=[btn])
        card.locator = MagicMock(return_value=btn)
        mock_page.locator = MagicMock(return_value=card)

        self.assertTrue(await bookmark_job(mock_page, "job-101"))
        self.assertTrue(await delete_job(mock_page, "job-101"))

    async def test_preview_and_execute_application(self):
        mock_page = MagicMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.keyboard.press = AsyncMock()

        btn = MockLocator(count_val=1, text="Apply")
        card = MockLocator(count_val=1, children=[btn])
        card.locator = MagicMock(return_value=btn)
        mock_page.locator = MagicMock(return_value=card)

        preview = await preview_application(mock_page, "job-101")
        self.assertEqual(preview.job_id, "job-101")
        self.assertEqual(preview.application_method, "direct_submission")

        executed = await execute_application(mock_page, "job-101")
        self.assertTrue(executed)


class TestApiClient(unittest.TestCase):
    """Tests for cache, keyword extraction, and job filtering."""

    def test_job_cache(self):
        # Test default TTL and attributes
        default_cache = JobCache()
        self.assertEqual(default_cache.ttl_seconds, 7200)
        self.assertEqual(default_cache.dismissed_ids, set())
        self.assertIsNotNone(default_cache._lock)

        cache = JobCache(ttl_minutes=1)
        self.assertEqual(cache.ttl_seconds, 60)
        self.assertTrue(cache.is_stale)
        self.assertEqual(len(cache.get_all()), 0)

        sample_jobs = [
            Job(job_id="j1", title="Backend Engineer", company="Acme", tech_stack=["Python", "FastAPI"]),
            Job(job_id="j2", title="Frontend Engineer", company="Beta", tech_stack=["React", "TypeScript"]),
        ]
        cache.update(sample_jobs)
        self.assertFalse(cache.is_stale)
        self.assertEqual(len(cache.get_all()), 2)
        self.assertEqual(cache.get_by_id("j1").title, "Backend Engineer")
        self.assertIsNone(cache.get_by_id("j999"))

        cache.clear()
        self.assertTrue(cache.is_stale)
        self.assertEqual(len(cache.get_all()), 0)

    def test_job_cache_ttl_env_config(self):
        """Test TTL configuration via CACHE_TTL_MINUTES environment variable."""
        with patch.dict(os.environ, {"CACHE_TTL_MINUTES": "45"}):
            cache = JobCache()
            self.assertEqual(cache.ttl_seconds, 45 * 60)

    def test_job_cache_dismiss_and_filtering(self):
        """Test dismissing jobs removes them from cache and prevents re-addition on update."""
        cache = JobCache()
        j1 = Job(job_id="j1", title="Backend Engineer", company="Acme")
        j2 = Job(job_id="j2", title="Frontend Engineer", company="Beta")
        j3 = Job(job_id="j3", title="DevOps Engineer", company="Gamma")

        cache.update([j1, j2])
        self.assertEqual(len(cache.get_all()), 2)
        self.assertIsNotNone(cache.get_by_id("j1"))

        # Dismiss j1
        cache.dismiss("j1")
        self.assertIn("j1", cache.dismissed_ids)
        self.assertIsNone(cache.get_by_id("j1"))
        self.assertEqual(len(cache.get_all()), 1)
        self.assertEqual(cache.get_all()[0].job_id, "j2")

        # Update with j1, j2, j3 - j1 should be filtered out
        cache.update([j1, j2, j3])
        self.assertEqual(len(cache.get_all()), 2)
        remaining_ids = [j.job_id for j in cache.get_all()]
        self.assertNotIn("j1", remaining_ids)
        self.assertIn("j2", remaining_ids)
        self.assertIn("j3", remaining_ids)

    def test_extract_cv_keywords_txt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("Experienced Senior Software Engineer skilled in Python, Kubernetes, PostgreSQL, and AWS.")
            temp_path = f.name

        try:
            keywords = extract_cv_keywords(temp_path)
            self.assertIn("Python", keywords)
            self.assertIn("Kubernetes", keywords)
            self.assertIn("PostgreSQL", keywords)
            self.assertIn("AWS", keywords)
        finally:
            os.unlink(temp_path)

    def test_extract_cv_keywords_docx(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            docx_path = f.name

        try:
            # Build valid minimal docx zip
            xml_content = b'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Proficient in TypeScript, React, Docker, and Redis.</w:t></w:r></w:p></w:body></w:document>'
            with zipfile.ZipFile(docx_path, "w") as z:
                z.writestr("word/document.xml", xml_content)

            keywords = extract_cv_keywords(docx_path)
            self.assertIn("TypeScript", keywords)
            self.assertIn("React", keywords)
            self.assertIn("Docker", keywords)
            self.assertIn("Redis", keywords)
        finally:
            os.unlink(docx_path)

    def test_filter_jobs(self):
        jobs = [
            Job(
                job_id="1",
                title="Senior Python Developer",
                company="Alpha",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["Python", "FastAPI", "Docker", "PostgreSQL"],
                description="Build scalable microservices with Python and FastAPI.",
            ),
            Job(
                job_id="2",
                title="Frontend Developer",
                company="Beta",
                location="New York, NY",
                work_mode=WorkMode.HYBRID,
                tech_stack=["React", "TypeScript", "Next.js"],
                description="Frontend UI development in Next.js.",
            ),
            Job(
                job_id="3",
                title="Legacy PHP Developer",
                company="Gamma",
                location="Remote",
                work_mode=WorkMode.REMOTE,
                tech_stack=["PHP", "MySQL"],
                description="Maintain legacy PHP codebase.",
            ),
        ]

        # 1. Test exclude keywords
        prefs = JobPreferences(
            tech_stack=["Python"],
            exclude_keywords=["PHP", "Legacy"],
        )
        filtered = filter_jobs(jobs, prefs)
        ids = [j.job_id for j in filtered]
        self.assertIn("1", ids)
        self.assertNotIn("3", ids)

        # 2. Test ranking and match score
        prefs_python = JobPreferences(
            tech_stack=["Python", "FastAPI", "Docker"],
            work_mode=WorkMode.REMOTE,
        )
        scored = filter_jobs(jobs, prefs_python)
        self.assertEqual(scored[0].job_id, "1")
        self.assertGreater(scored[0].match_score, 80.0)

        # 3. Test explainability fields populated on job 1
        top_job = scored[0]
        self.assertIn("Python", top_job.matched_skills)
        self.assertIn("FastAPI", top_job.matched_skills)
        self.assertIn("Docker", top_job.matched_skills)
        self.assertEqual(top_job.missing_skills, [])
        self.assertEqual(top_job.seniority_level, "Senior")
        self.assertIsNotNone(top_job.description_summary)
        self.assertTrue(len(top_job.match_reasons) > 0)
        self.assertTrue(any("stack matched" in r.lower() for r in top_job.match_reasons))

    def test_job_schema_explainability_defaults_and_serialization(self):
        """Test Job schema defaults for new explainability fields and serialization."""
        job = Job(job_id="test-1", title="Backend Engineer", company="TestCo")
        self.assertEqual(job.matched_skills, [])
        self.assertEqual(job.missing_skills, [])
        self.assertEqual(job.match_reasons, [])
        self.assertIsNone(job.description_summary)
        self.assertIsNone(job.seniority_level)

        dumped = job.model_dump()
        self.assertIn("matched_skills", dumped)
        self.assertIn("missing_skills", dumped)
        self.assertIn("match_reasons", dumped)
        self.assertIn("description_summary", dumped)
        self.assertIn("seniority_level", dumped)

    def test_detect_seniority_level_titles(self):
        """Test seniority detection across different title variations."""
        from job_mcp.core.api_client import detect_seniority_level

        # Junior variations
        self.assertEqual(detect_seniority_level("Junior Python Developer"), "Junior")
        self.assertEqual(detect_seniority_level("Entry Level Backend Engineer"), "Junior")
        self.assertEqual(detect_seniority_level("Graduate Software Engineer"), "Junior")

        # Student / Intern variations
        self.assertEqual(detect_seniority_level("Student Software Developer"), "Student")
        self.assertEqual(detect_seniority_level("Software Engineering Intern"), "Intern")
        self.assertEqual(detect_seniority_level("Summer Internship - Dev"), "Intern")

        # Senior variations
        self.assertEqual(detect_seniority_level("Senior Fullstack Developer"), "Senior")
        self.assertEqual(detect_seniority_level("Sr. Backend Engineer"), "Senior")
        self.assertEqual(detect_seniority_level("Sr Software Engineer"), "Senior")
        self.assertEqual(detect_seniority_level("Principal DevOps Engineer"), "Senior")
        self.assertEqual(detect_seniority_level("Staff Infrastructure Engineer"), "Senior")
        self.assertEqual(detect_seniority_level("Cloud Solutions Architect"), "Senior")

        # Lead variations
        self.assertEqual(detect_seniority_level("Tech Lead"), "Lead")
        self.assertEqual(detect_seniority_level("Engineering Team Lead"), "Lead")
        self.assertEqual(detect_seniority_level("Head of Engineering"), "Lead")
        self.assertEqual(detect_seniority_level("Director of R&D"), "Lead")
        self.assertEqual(detect_seniority_level("VP Engineering"), "Lead")

        # Mid variations
        self.assertEqual(detect_seniority_level("Mid Python Developer"), "Mid")
        self.assertEqual(detect_seniority_level("Intermediate Frontend Engineer"), "Mid")

        # Unspecified
        self.assertIsNone(detect_seniority_level("Software Engineer"))

    def test_detect_seniority_level_text_fallback(self):
        """Test seniority detection falls back to description text when title is neutral."""
        from job_mcp.core.api_client import detect_seniority_level

        self.assertEqual(
            detect_seniority_level("Software Engineer", "This is an entry-level position for fast learners."),
            "Junior"
        )
        self.assertEqual(
            detect_seniority_level("Software Engineer", "Looking for a student in computer science."),
            "Student"
        )
        self.assertEqual(
            detect_seniority_level("Backend Developer", "Role as a senior engineer mentoring junior members."),
            "Senior"
        )

    def test_generate_description_summary(self):
        """Test description summary generator logic."""
        from job_mcp.core.api_client import generate_description_summary

        self.assertIsNone(generate_description_summary(""))
        self.assertIsNone(generate_description_summary("   "))

        short_desc = "Build next-gen cloud platforms with Python. Collaborate with international team. Excellent compensation."
        summary = generate_description_summary(short_desc, max_chars=150)
        self.assertIsNotNone(summary)
        self.assertIn("Build next-gen", summary)
        self.assertTrue(len(summary) <= 150)

    def test_filter_jobs_junior_with_senior_in_body_not_disqualified(self):
        """Test Junior job mentioning senior in description is NOT excluded when exclude_keywords=['Senior']."""
        junior_job = Job(
            job_id="j-1",
            title="Junior Python Developer",
            company="StartupCo",
            tech_stack=["Python", "FastAPI"],
            description="You will be mentored by senior engineers and architects on high-load services.",
        )
        senior_job = Job(
            job_id="s-1",
            title="Senior Python Architect",
            company="BigCorp",
            tech_stack=["Python", "FastAPI"],
            description="Lead architecture and mentor developers.",
        )

        prefs = JobPreferences(
            tech_stack=["Python", "FastAPI"],
            exclude_keywords=["Senior", "Architect"],
        )

        results = filter_jobs([junior_job, senior_job], prefs)
        result_ids = [j.job_id for j in results]

        self.assertIn("j-1", result_ids)
        self.assertNotIn("s-1", result_ids)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].seniority_level, "Junior")
        self.assertIn("Python", results[0].matched_skills)
        self.assertIn("FastAPI", results[0].matched_skills)

    def test_resolve_cv_path_exact_path(self):
        """Test resolve_cv_path resolves an explicit existing file path."""
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as f:
            f.write("mock pdf content")
            temp_path = f.name

        try:
            resolved = resolve_cv_path(temp_path)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved, Path(temp_path).resolve())
        finally:
            os.unlink(temp_path)

    def test_resolve_cv_path_env_var(self):
        """Test resolve_cv_path resolves via DEFAULT_CV_PATH environment variable."""
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as f:
            f.write("mock pdf content")
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"DEFAULT_CV_PATH": temp_path}):
                resolved = resolve_cv_path()
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved, Path(temp_path).resolve())
        finally:
            os.unlink(temp_path)

    def test_resolve_cv_path_fallback_workspace(self):
        """Test resolve_cv_path finds workspace CV files when cv_path is omitted and env is empty."""
        with patch.dict(os.environ, {}, clear=True):
            resolved = resolve_cv_path()
            if (Path.cwd() / "lior_zvieli_cv.pdf").is_file():
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved, (Path.cwd() / "lior_zvieli_cv.pdf").resolve())

    def test_resolve_cv_path_container_fallback(self):
        """Test resolve_cv_path resolves container paths when running inside container."""
        def mock_is_file(path_obj):
            return str(path_obj) == "/app/lior_zvieli_cv.pdf"

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "is_file", autospec=True, side_effect=lambda self: str(self) == "/app/lior_zvieli_cv.pdf"):
                with patch.object(Path, "resolve", autospec=True, side_effect=lambda self: self):
                    resolved = resolve_cv_path()
                    self.assertIsNotNone(resolved)
                    self.assertEqual(str(resolved), "/app/lior_zvieli_cv.pdf")

    def test_resolve_cv_path_nonexistent_returns_none(self):
        """Test resolve_cv_path returns None when no candidates exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {}, clear=True):
                    resolved = resolve_cv_path("nonexistent_path_xyz.pdf")
                    self.assertIsNone(resolved)

                    resolved_empty = resolve_cv_path()
                    self.assertIsNone(resolved_empty)
            finally:
                os.chdir(orig_cwd)

    def test_extract_cv_keywords_modern_ai_web3_backend(self):
        """Test extract_cv_keywords correctly detects new AI, Web3, and Modern Framework keywords."""
        sample_text = (
            "Senior AI & Blockchain Engineer proficient in GraphRAG, LangGraph, RAG, Agentic workflows, "
            "Vector DB, ChromaDB, Chroma, Pinecone, Qdrant, Weaviate, CrewAI, Autogen, vLLM, Ollama, "
            "LangSmith, Semantic Kernel, Transformers, Fine-Tuning, and Embeddings. "
            "Deep expertise in Web3, Solidity, Smart Contracts, EVM, Hardhat, Foundry, Ethers.js, Viem, "
            "FastAPI, Supabase, Pydantic, SQLAlchemy, Prisma, TRPC, and AsyncIO."
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(sample_text)
            temp_path = f.name

        try:
            keywords = extract_cv_keywords(temp_path)
            expected_keywords = [
                # AI / LLM / Agentic
                "GraphRAG", "LangGraph", "RAG", "Agentic", "Vector DB", "ChromaDB", "Chroma",
                "Pinecone", "Qdrant", "Weaviate", "CrewAI", "Autogen", "vLLM", "Ollama",
                "LangSmith", "Semantic Kernel", "Transformers", "Fine-Tuning", "Embeddings",
                # Web3 & Blockchain
                "Solidity", "Web3", "Smart Contracts", "EVM", "Hardhat", "Foundry", "Ethers.js", "Viem",
                # Modern Backend & Frameworks
                "FastAPI", "Supabase", "Pydantic", "SQLAlchemy", "Prisma", "TRPC", "AsyncIO",
            ]
            for kw in expected_keywords:
                self.assertIn(kw, keywords, f"Expected keyword '{kw}' was not extracted from CV.")
        finally:
            os.unlink(temp_path)

    def test_extract_cv_keywords_default_resolution_and_nonexistent(self):
        """Test extract_cv_keywords with default resolution and nonexistent path handling."""
        # 1. Nonexistent path handling
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {}, clear=True):
                    res = extract_cv_keywords("nonexistent.pdf")
                    self.assertEqual(res, [])
            finally:
                os.chdir(orig_cwd)

        # 2. Real workspace CV extraction (if lior_zvieli_cv.pdf exists)
        if (Path.cwd() / "lior_zvieli_cv.pdf").is_file():
            keywords = extract_cv_keywords()
            self.assertIsInstance(keywords, list)
            self.assertGreater(len(keywords), 0)

    def test_filter_jobs_cv_scoring_tuning_and_coverage(self):
        """Test CV scoring formula does not penalize rich CVs and computes job coverage properly."""
        # Candidate with broad AI/Fullstack CV with 25+ skills
        cv_text = (
            "Senior AI & Fullstack Developer. Skills: Python, Docker, LangChain, LLM, FastAPI, "
            "React, TypeScript, Kubernetes, SQL, Git, Linux, AWS, Redis, PostgreSQL, GraphQL, "
            "Next.js, C++, PyTorch, Pandas, NumPy, Terraform, Supabase, SQLAlchemy, Pydantic, AsyncIO."
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(cv_text)
            cv_path = f.name

        try:
            ai_job = Job(
                job_id="ai-1",
                title="AI Engineer",
                company="BrainAI",
                tech_stack=["Python", "LangChain", "LLM", "Docker"],
                description="Developing LLM applications using LangChain, Python, and Docker.",
            )
            fullstack_job = Job(
                job_id="fs-1",
                title="Fullstack Developer",
                company="WebCo",
                tech_stack=["React", "TypeScript", "FastAPI", "PostgreSQL"],
                description="Build modern web applications with React, TypeScript, and FastAPI.",
            )
            irrelevant_job = Job(
                job_id="irr-1",
                title="Help Desk Support Technician",
                company="SupportCo",
                tech_stack=[],
                description="Provide telephone and hardware support for internal office staff.",
            )
            finance_job = Job(
                job_id="fin-1",
                title="Finance Analyst",
                company="BankCorp",
                tech_stack=[],
                description="Financial modeling, budget forecasting, and accounting spreadsheets.",
            )

            prefs = JobPreferences(cv_path=cv_path)
            ranked = filter_jobs([ai_job, fullstack_job, irrelevant_job, finance_job], prefs)

            self.assertEqual(len(ranked), 4)
            scored_map = {j.job_id: j for j in ranked}

            # 1. AI job with 3+ matching skills (Python, LangChain, LLM, Docker) scores >= 75.0 (between 75.0 and 98.0+)
            ai_scored = scored_map["ai-1"]
            self.assertGreaterEqual(ai_scored.match_score, 75.0)
            self.assertLessEqual(ai_scored.match_score, 100.0)
            self.assertIn("Python", ai_scored.matched_skills)
            self.assertIn("LangChain", ai_scored.matched_skills)
            self.assertIn("LLM", ai_scored.matched_skills)
            self.assertIn("Docker", ai_scored.matched_skills)

            # 2. Fullstack job with 3+ matching skills scores >= 75.0
            fs_scored = scored_map["fs-1"]
            self.assertGreaterEqual(fs_scored.match_score, 75.0)
            self.assertLessEqual(fs_scored.match_score, 100.0)

            # 3. Irrelevant / Finance jobs with 0 matching skills score < 40.0 (specifically < 35.0)
            irr_scored = scored_map["irr-1"]
            fin_scored = scored_map["fin-1"]
            self.assertLess(irr_scored.match_score, 40.0)
            self.assertLess(irr_scored.match_score, 35.0)
            self.assertEqual(irr_scored.match_score, 0.0)
            self.assertLess(fin_scored.match_score, 40.0)
            self.assertLess(fin_scored.match_score, 35.0)
            self.assertEqual(fin_scored.match_score, 0.0)

            # 4. Explainable match reasons on AI job
            self.assertTrue(len(ai_scored.match_reasons) > 0)
            cv_reason = next((r for r in ai_scored.match_reasons if "CV matched" in r), None)
            self.assertIsNotNone(cv_reason)
            self.assertIn("core skills:", cv_reason)
            self.assertIn("tech requirements match", cv_reason)
            self.assertIn("100% tech requirements match", cv_reason)
        finally:
            os.unlink(cv_path)

    def test_filter_jobs_compatibility_with_explicit_tech_stack_and_keywords(self):
        """Test scoring combines explicit tech_stack (50%), keywords (30%), and CV (20%)."""
        cv_text = "Developer skilled in Python, Docker, Redis, Git, Linux."
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(cv_text)
            cv_path = f.name

        try:
            job_match_all = Job(
                job_id="j-all",
                title="Python Platform Engineer",
                company="CloudTech",
                tech_stack=["Python", "FastAPI", "Docker", "Redis"],
                description="Scale platform microservices with Python and FastAPI on Kubernetes.",
            )
            job_no_match = Job(
                job_id="j-none",
                title="HR Manager",
                company="PeopleCo",
                tech_stack=[],
                description="Manage recruitment and people operations.",
            )

            prefs = JobPreferences(
                tech_stack=["Python", "FastAPI"],
                keywords=["Kubernetes"],
                cv_path=cv_path,
            )

            results = filter_jobs([job_match_all, job_no_match], prefs)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].job_id, "j-all")
            self.assertGreaterEqual(results[0].match_score, 85.0)

            reasons = results[0].match_reasons
            self.assertTrue(any("CV matched" in r for r in reasons))
            self.assertTrue(any("Target stack matched" in r for r in reasons))
            self.assertTrue(any("Keywords matched" in r for r in reasons))

            self.assertEqual(results[1].job_id, "j-none")
            self.assertEqual(results[1].match_score, 0.0)
            self.assertEqual(results[1].match_reasons, [])
        finally:
            os.unlink(cv_path)

    def test_extract_dynamic_cv_skills_section_based(self):
        """Test extract_dynamic_cv_skills with structured section-based skills text."""
        section_text = """
Technical Skills
•Languages:Python, Rust, Solidity, Noir, JavaScript, TypeScript, C, C#
•AI & Data Engineering:GraphRAG, LangGraph, LangChain, Azure AI Search, Azure Document Intelligence,
Hugging Face, Scikit-Learn, NetworkX, Pandas, Ollama
•Web & Decentralized:React, Node.js, Foundry, Hardhat, Ethers.js, Viem, IPFS, Smart Contract Development
•DevOps & Cloud:Azure (Container Apps, Cosmos DB, Azure Functions, Blob Storage), Docker, Linux, Git
"""
        skills = extract_dynamic_cv_skills(section_text)
        expected_skills = [
            "Python", "Rust", "Solidity", "Noir", "JavaScript", "TypeScript", "C", "C#",
            "GraphRAG", "LangGraph", "LangChain", "Azure AI Search", "Azure Document Intelligence",
            "Hugging Face", "Scikit-Learn", "NetworkX", "Pandas", "Ollama",
            "React", "Node.js", "Foundry", "Hardhat", "Ethers.js", "Viem", "IPFS",
            "Smart Contract Development", "Azure", "Container Apps", "Cosmos DB",
            "Azure Functions", "Blob Storage", "Docker", "Linux", "Git"
        ]
        for skill in expected_skills:
            self.assertIn(skill, skills, f"Expected skill '{skill}' was not extracted from section.")

    def test_extract_dynamic_cv_skills_bulleted_project_descriptions(self):
        """Test extract_dynamic_cv_skills with unstructured bulleted project descriptions."""
        project_text = """
Projects
ZK Credit Agent – Privacy-Preserving Credit Oracle
• Architected a decentralized credit oracle bridging verified Ethereum Mainnet account history to L2s, utilizing Axiom V3 for historical query orchestration and Noir ZK circuits for privacy-preserving computations.
• Developed Noir circuits to verify Merkle Patricia Trie (MPT) membership proofs, validating storage slots against Mainnet state roots using custom Keccak256 hash constraints.
• Authored Solidity smart contracts utilizing Foundry for rigorous testing, establishing a self-sustaining prover marketplace.
• Engineered a TypeScript prover agent integrating Viem and Node.js to monitor on-chain requests, automate UltraHonk proof generation (bb.js).
• Built a React-based frontend powered by Vite and TailwindCSS, incorporating single-signature session key architectures.
• Developed a multi-label classification system for telemedicine utilizing ClinicalBERT and Hugging Face.
• Engineered a synthetic data generation pipeline using Ollama (LLMs) to mitigate class imbalance.
"""
        skills = extract_dynamic_cv_skills(project_text)
        expected_skills = [
            "ZK", "Noir", "Axiom", "Axiom V3", "MPT", "Merkle Patricia Trie", "Keccak256",
            "Solidity", "Smart Contracts", "Foundry", "TypeScript", "Viem", "Node.js",
            "UltraHonk", "bb.js", "React", "Vite", "TailwindCSS", "ClinicalBERT",
            "Hugging Face", "Ollama", "LLM"
        ]
        for skill in expected_skills:
            self.assertIn(skill, skills, f"Expected skill '{skill}' was not extracted from project text.")

    def test_extract_dynamic_cv_skills_stopword_rejection(self):
        """Test stopword rejection ensures resume structural words, dates, and noise are filtered out."""
        noisy_text = """
Lior Zvieli
liorzvieli@gmail.com — LinkedIn — GitHub — +972-52-2276810
Professional Summary
Computer Science student at HIT with a strong background in engineering high-performance decentralized protocols.
March 2026 – Present, Summer 2025, Winter, Fall, Spring
Education
B.Sc. in Computer Science, Holon Institute of Technology (HIT), Degree, Expected: Summer 2026
Experience at Client Project MAG Corps IDF, Senior Architect & Developer
Key Courses: 100, 94, 92, 87, 0.8
"""
        skills = extract_dynamic_cv_skills(noisy_text)
        forbidden_stopwords = [
            "LinkedIn", "March", "Education", "Summary", "Hit", "Developer", "Architect",
            "Engineer", "BSc", "Computer", "Science", "Professional", "Student", "Israel",
            "Summer", "Winter", "Fall", "Spring", "Present", "Experience", "Projects",
            "Client", "Project", "Systems", "Expected", "Courses", "Legal", "Hebrew",
            "Mainnet", "Key", "Github", "Email", "Phone", "University", "College", "Degree",
            "Idf", "Work", "Strong", "Background", "Script", "Level"
        ]
        for word in forbidden_stopwords:
            self.assertNotIn(word, skills, f"Stopword '{word}' should have been filtered out but was present in: {skills}")
            self.assertNotIn(word.upper(), skills, f"Stopword '{word.upper()}' should have been filtered out.")

    def test_extract_cv_keywords_lior_zvieli_pdf(self):
        """Test extract_cv_keywords extracts 40+ dynamic skills from real lior_zvieli_cv.pdf."""
        cv_path = Path("lior_zvieli_cv.pdf")
        if not cv_path.is_file():
            self.skipTest("lior_zvieli_cv.pdf not present in workspace")

        keywords = extract_cv_keywords(str(cv_path))
        self.assertGreaterEqual(len(keywords), 40, f"Expected 40+ skills, got {len(keywords)}: {keywords}")

        expected_skills = [
            "GraphRAG", "LangGraph", "LangChain", "LlamaIndex", "FastAPI",
            "AI Search", "Azure", "NetworkX", "Ollama", "React", "Node.js",
            "Docker", "Linux", "Git", "Python", "TypeScript", "JavaScript",
            "Solidity", "Noir", "Foundry", "TailwindCSS", "Vite", "ZK",
            "Scikit-Learn", "Hugging Face", "Elasticsearch", "SQL"
        ]
        for s in expected_skills:
            self.assertIn(s, keywords, f"Expected skill '{s}' not found in extracted CV keywords.")


if __name__ == "__main__":
    unittest.main()
