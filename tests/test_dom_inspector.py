"""Unit tests for Universal DOM Inspector & Schema Extractor."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock
from playwright.async_api import async_playwright

from job_mcp.core.application.dom_inspector import (
    FormFieldSchema,
    SubmitButtonInfo,
    _disambiguate_submit_button_with_llm,
    _score_button_candidate,
    extract_form_schema,
    identify_submit_button,
)


class TestDOMInspectorHeuristics(unittest.TestCase):
    """Unit tests for deterministic scoring and schema validation."""

    def test_score_button_candidate_heuristics(self):
        """Test heuristic scoring unit logic."""
        # High score for explicit submit
        score, action = _score_button_candidate(
            {"text": "Submit Application", "button_type": "submit", "element_id": "submit_btn"}
        )
        self.assertGreaterEqual(score, 120)
        self.assertEqual(action, "submit")

        # Next / continue
        score, action = _score_button_candidate({"text": "Next Step", "button_type": "button"})
        self.assertGreaterEqual(score, 60)
        self.assertEqual(action, "continue")

        # Negative exclusion
        score, action = _score_button_candidate({"text": "Cancel", "button_type": "button"})
        self.assertLess(score, 0)
        self.assertEqual(action, "ignore")

        score, action = _score_button_candidate({"text": "Sign in to apply", "button_type": "button"})
        self.assertLess(score, 0)

    def test_form_field_schema_serialization(self):
        """Test FormFieldSchema model dump."""
        field = FormFieldSchema(
            field_id="first_name",
            name="candidate_first_name",
            label="First Name *",
            field_type="text",
            required=True,
            options=None,
            frame_index=0,
            selector="#first_name",
        )
        d = field.to_dict()
        self.assertEqual(d["field_id"], "first_name")
        self.assertEqual(d["name"], "candidate_first_name")
        self.assertTrue(d["required"])
        self.assertEqual(d["field_type"], "text")


class TestDOMInspectorAsync(unittest.IsolatedAsyncioTestCase):
    """Asynchronous DOM inspection tests with Playwright and mock frames."""

    async def asyncSetUp(self):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    async def asyncTearDown(self):
        await self.context.close()
        await self.browser.close()
        await self.pw.stop()

    async def test_extract_form_schema_standard_fields(self):
        """Test extracting diverse form fields: text, email, tel, file, select, textarea, checkbox."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Job Application</title></head>
        <body>
            <form id="job-app-form">
                <div class="form-group">
                    <label for="full_name">Full Name *</label>
                    <input type="text" id="full_name" name="candidate_name" required placeholder="John Doe" />
                </div>

                <div class="form-group">
                    <label for="email_addr">Email Address</label>
                    <input type="email" id="email_addr" name="email" required />
                </div>

                <div class="form-group">
                    <label for="phone_num">Phone Number</label>
                    <input type="tel" id="phone_num" name="phone" placeholder="+972-50-1234567" />
                </div>

                <div class="form-group">
                    <label for="resume_file">Upload CV / Resume</label>
                    <input type="file" id="resume_file" name="resume" required />
                </div>

                <div class="form-group">
                    <label for="experience_lvl">Years of Experience</label>
                    <select id="experience_lvl" name="experience">
                        <option value="">Please Select...</option>
                        <option value="junior">0-2 Years</option>
                        <option value="mid">3-5 Years</option>
                        <option value="senior">6+ Years</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="cover_letter">Cover Letter / Notes</label>
                    <textarea id="cover_letter" name="notes" placeholder="Tell us why you are a great fit..."></textarea>
                </div>

                <div class="form-group">
                    <label>
                        <input type="checkbox" id="work_auth" name="israel_work_auth" required />
                        I am legally authorized to work in Israel
                    </label>
                </div>

                <!-- Hidden field that should be excluded -->
                <input type="hidden" name="csrf_token" value="abc123xyz" />
            </form>
        </body>
        </html>
        """
        await self.page.set_content(html_content)

        fields = await extract_form_schema(self.page)
        self.assertEqual(len(fields), 7)

        field_map = {f.field_id: f for f in fields}

        # 1. Full Name
        name_field = field_map["full_name"]
        self.assertEqual(name_field.name, "candidate_name")
        self.assertIn("Full Name", name_field.label)
        self.assertEqual(name_field.field_type, "text")
        self.assertTrue(name_field.required)
        self.assertEqual(name_field.placeholder, "John Doe")

        # 2. Email Address
        email_field = field_map["email_addr"]
        self.assertEqual(email_field.name, "email")
        self.assertIn("Email Address", email_field.label)
        self.assertEqual(email_field.field_type, "email")
        self.assertTrue(email_field.required)

        # 3. Phone
        phone_field = field_map["phone_num"]
        self.assertEqual(phone_field.name, "phone")
        self.assertIn("Phone Number", phone_field.label)
        self.assertEqual(phone_field.field_type, "tel")
        self.assertFalse(phone_field.required)

        # 4. Resume File Upload
        file_field = field_map["resume_file"]
        self.assertEqual(file_field.name, "resume")
        self.assertTrue("Resume" in file_field.label or "CV" in file_field.label)
        self.assertEqual(file_field.field_type, "file")
        self.assertTrue(file_field.required)

        # 5. Experience Select
        select_field = field_map["experience_lvl"]
        self.assertEqual(select_field.name, "experience")
        self.assertEqual(select_field.field_type, "select")
        self.assertEqual(select_field.options, ["0-2 Years", "3-5 Years", "6+ Years"])

        # 6. Cover Letter Textarea
        ta_field = field_map["cover_letter"]
        self.assertEqual(ta_field.name, "notes")
        self.assertEqual(ta_field.field_type, "textarea")
        self.assertIn("Cover Letter", ta_field.label)

        # 7. Checkbox
        cb_field = field_map["work_auth"]
        self.assertEqual(cb_field.name, "israel_work_auth")
        self.assertEqual(cb_field.field_type, "checkbox")
        self.assertIn("authorized to work in Israel", cb_field.label)
        self.assertTrue(cb_field.required)

        # Confirm hidden field excluded
        self.assertNotIn("csrf_token", field_map)

    async def test_extract_form_schema_label_heuristics(self):
        """Test aria-label, aria-labelledby, placeholder fallback, preceding text, and fieldset legend."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <!-- 1. aria-label -->
            <input type="text" id="github_in" aria-label="GitHub Profile URL" />

            <!-- 2. aria-labelledby -->
            <span id="salary_lbl">Expected Annual Compensation</span>
            <input type="number" id="salary_in" aria-labelledby="salary_lbl" />

            <!-- 3. Preceding sibling label -->
            <label>LinkedIn URL</label>
            <input type="url" id="linkedin_in" />

            <!-- 4. Placeholder fallback when no label -->
            <input type="text" id="portfolio_in" placeholder="Portfolio / Personal Website" />

            <!-- 5. Fieldset legend for radio group -->
            <fieldset>
                <legend>Work Preference</legend>
                <label><input type="radio" name="work_mode" value="hybrid" /> Hybrid</label>
                <label><input type="radio" name="work_mode" value="remote" /> Remote</label>
                <label><input type="radio" name="work_mode" value="onsite" /> On-site</label>
            </fieldset>
        </body>
        </html>
        """
        await self.page.set_content(html_content)

        fields = await extract_form_schema(self.page)
        field_map = {f.field_id: f for f in fields}

        # 1. aria-label
        self.assertEqual(field_map["github_in"].label, "GitHub Profile URL")
        self.assertEqual(field_map["github_in"].field_type, "text")

        # 2. aria-labelledby
        self.assertEqual(field_map["salary_in"].label, "Expected Annual Compensation")
        self.assertEqual(field_map["salary_in"].field_type, "number")

        # 3. Preceding sibling
        self.assertEqual(field_map["linkedin_in"].label, "LinkedIn URL")
        self.assertEqual(field_map["linkedin_in"].field_type, "url")

        # 4. Placeholder fallback
        self.assertEqual(field_map["portfolio_in"].label, "Portfolio / Personal Website")

        # 5. Radio group
        radio_group = next(f for f in fields if f.field_type == "radio")
        self.assertEqual(radio_group.name, "work_mode")
        self.assertIn("Hybrid", radio_group.options)
        self.assertIn("Remote", radio_group.options)
        self.assertIn("On-site", radio_group.options)

    async def test_extract_form_schema_nested_iframes(self):
        """Test extracting form schema across main frame and child iframes."""
        main_html = """
        <!DOCTYPE html>
        <html>
        <body>
            <h1>Company Careers Portal</h1>
            <input type="text" id="main_search" placeholder="Search Jobs" />

            <iframe id="app-iframe" srcdoc='
                <html>
                <body>
                    <form id="embedded-app">
                        <label for="if_name">Applicant Name</label>
                        <input type="text" id="if_name" name="full_name" required />

                        <label for="if_email">Applicant Email</label>
                        <input type="email" id="if_email" name="email_address" required />

                        <label for="if_cv">Upload Resume</label>
                        <input type="file" id="if_cv" name="cv_upload" required />

                        <button type="submit" id="if_submit">Submit Application</button>
                    </form>
                </body>
                </html>
            '></iframe>
        </body>
        </html>
        """
        await self.page.set_content(main_html)
        await self.page.wait_for_timeout(300)

        fields = await extract_form_schema(self.page)
        self.assertGreaterEqual(len(fields), 4)

        main_fields = [f for f in fields if f.frame_index == 0]
        iframe_fields = [f for f in fields if f.frame_index > 0]

        self.assertGreaterEqual(len(main_fields), 1)
        self.assertTrue(any(f.field_id == "main_search" for f in main_fields))

        self.assertGreaterEqual(len(iframe_fields), 3)
        iframe_map = {f.field_id: f for f in iframe_fields}
        self.assertIn("if_name", iframe_map)
        self.assertTrue(iframe_map["if_name"].required)
        self.assertIn("if_email", iframe_map)
        self.assertIn("if_cv", iframe_map)

    async def test_identify_submit_button_heuristics(self):
        """Test deterministic identification of submit buttons while ignoring cancel/back/search buttons."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <form>
                <input type="text" name="name" />
                <div class="actions">
                    <button type="button" class="btn-secondary" id="btn_cancel">Cancel Application</button>
                    <button type="button" class="btn-back" id="btn_back">Previous Step</button>
                    <button type="submit" class="btn-primary" id="btn_submit">Submit Application</button>
                </div>
            </form>
        </body>
        </html>
        """
        await self.page.set_content(html_content)

        btn = await identify_submit_button(self.page)
        self.assertIsNotNone(btn)
        self.assertEqual(btn.action_type, "submit")
        self.assertIn("Submit Application", btn.text)
        self.assertEqual(btn.selector, "#btn_submit")
        self.assertGreaterEqual(btn.confidence, 0.8)

    async def test_identify_submit_button_multi_step_flow(self):
        """Test identifying multi-step continue/next buttons."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <form>
                <input type="text" name="first_name" />
                <button type="button" id="btn_next" class="btn primary">Save & Continue</button>
            </form>
        </body>
        </html>
        """
        await self.page.set_content(html_content)

        btn = await identify_submit_button(self.page)
        self.assertIsNotNone(btn)
        self.assertEqual(btn.action_type, "continue")
        self.assertIn("Save & Continue", btn.text)
        self.assertEqual(btn.selector, "#btn_next")

    async def test_identify_submit_button_in_iframe(self):
        """Test locating submit button when inside an iframe."""
        main_html = """
        <!DOCTYPE html>
        <html>
        <body>
            <button id="main_menu">Menu</button>
            <iframe srcdoc='
                <html>
                <body>
                    <form>
                        <input type="text" name="test" />
                        <input type="submit" id="sub_input" value="Send Application" />
                    </form>
                </body>
                </html>
            '></iframe>
        </body>
        </html>
        """
        await self.page.set_content(main_html)
        await self.page.wait_for_timeout(300)

        btn = await identify_submit_button(self.page)
        self.assertIsNotNone(btn)
        self.assertGreater(btn.frame_index, 0)
        self.assertIn("Send Application", btn.text)

        locator = btn.get_locator(self.page)
        self.assertIsNotNone(locator)
        count = await locator.count()
        self.assertEqual(count, 1)

    async def test_disambiguate_submit_button_with_llm(self):
        """Test LLM gateway disambiguation for ambiguous buttons."""
        candidates = [
            {"text": "Proceed to Review", "button_type": "button", "selector": "#btn1", "element_class": "btn"},
            {"text": "Save Draft", "button_type": "button", "selector": "#btn2", "element_class": "btn"},
        ]

        mock_llm = AsyncMock()
        mock_llm.ask_question.return_value = "The best button is index 0 (Proceed to Review)."

        idx = await _disambiguate_submit_button_with_llm(candidates, mock_llm)
        self.assertEqual(idx, 0)
        mock_llm.ask_question.assert_called_once()

    async def test_identify_submit_button_with_mock_llm_gateway(self):
        """Test identify_submit_button triggering LLM when ambiguous."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <form>
                <button type="button" id="btn_step2" class="step-btn">Continue to Review</button>
                <button type="button" id="btn_next" class="step-btn">Next Step</button>
            </form>
        </body>
        </html>
        """
        await self.page.set_content(html_content)

        mock_llm = AsyncMock()
        mock_llm.ask_question.return_value = "1"

        btn = await identify_submit_button(self.page, llm_gateway=mock_llm)
        self.assertIsNotNone(btn)
        self.assertIn(btn.text, ("Continue to Review", "Next Step"))

    async def test_extract_form_schema_empty_page(self):
        """Test extracting from a page without form elements returns empty list."""
        await self.page.set_content("<div><h1>Just a static article</h1><p>No forms here.</p></div>")

        fields = await extract_form_schema(self.page)
        self.assertEqual(fields, [])

        btn = await identify_submit_button(self.page)
        self.assertIsNone(btn)

    async def test_extract_form_schema_with_mock_detached_frame(self):
        """Test handling detached or failing frames without crashing."""
        mock_page = MagicMock()
        mock_frame1 = AsyncMock()
        mock_frame1.is_detached.return_value = True

        mock_frame2 = AsyncMock()
        mock_frame2.is_detached.return_value = False
        mock_frame2.evaluate.side_effect = Exception("Frame execution context destroyed")

        mock_page.frames = [mock_frame1, mock_frame2]

        fields = await extract_form_schema(mock_page)
        self.assertEqual(fields, [])

        btn = await identify_submit_button(mock_page)
        self.assertIsNone(btn)
