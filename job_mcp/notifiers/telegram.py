"""Telegram bot notifier for job alert delivery."""

from __future__ import annotations

import asyncio
import html
import logging
import os
from typing import Optional

import httpx

from job_mcp.models.schemas import Job
from job_mcp.notifiers.base import BaseNotifier

logger = logging.getLogger(__name__)


class TelegramNotifier(BaseNotifier):
    """Notifier for dispatching job alerts to Telegram chats or channels."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        parse_mode: str = "HTML",
        timeout: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
        api_base_url: str = "https://api.telegram.org",
        max_message_length: int = 4096,
        max_retries: int = 2,
    ) -> None:
        """Initialize TelegramNotifier.

        Args:
            bot_token: Telegram bot API token (or reads TELEGRAM_BOT_TOKEN from env).
            chat_id: Telegram chat ID or channel username (or reads TELEGRAM_CHAT_ID from env).
            parse_mode: Telegram formatting mode ("HTML" or "Markdown"). Default is "HTML".
            timeout: HTTP request timeout in seconds.
            client: Optional reusable httpx.AsyncClient.
            api_base_url: Telegram API base URL.
            max_message_length: Maximum allowed characters per Telegram message (default 4096).
            max_retries: Maximum retries for rate-limited requests (429).
        """
        self.bot_token: str = bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id: str = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")
        self.parse_mode: str = parse_mode
        self.timeout: float = timeout
        self._client: Optional[httpx.AsyncClient] = client
        self.api_base_url: str = api_base_url.rstrip("/")
        self.max_message_length: int = max_message_length
        self.max_retries: int = max_retries

    @property
    def is_configured(self) -> bool:
        """Check whether bot token and chat ID are configured."""
        return bool(self.bot_token and self.chat_id)

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create an AsyncClient."""
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=self.timeout)

    def _format_job_html(self, job: Job) -> str:
        """Format a single Job object into Telegram HTML markup."""
        title = html.escape(job.title or "Untitled Position")
        company = html.escape(job.company or "Unknown Company")
        url = job.apply_url or job.url

        if url:
            header = f"💼 <b><a href=\"{html.escape(url)}\">{title}</a></b> @ <b>{company}</b>"
        else:
            header = f"💼 <b>{title}</b> @ <b>{company}</b>"

        details: list[str] = []

        if job.location or job.work_mode:
            loc_parts = []
            if job.location:
                loc_parts.append(f"📍 {html.escape(job.location)}")
            if job.work_mode:
                mode_str = job.work_mode.value if hasattr(job.work_mode, "value") else str(job.work_mode)
                loc_parts.append(f"🏢 {html.escape(mode_str.capitalize())}")
            details.append(" | ".join(loc_parts))

        if job.match_score is not None:
            details.append(f"🎯 <b>Match Score:</b> {job.match_score:.1f}%")

        if job.salary_range:
            details.append(f"💰 <b>Salary:</b> {html.escape(job.salary_range)}")

        if job.tech_stack:
            tech_str = ", ".join(html.escape(t) for t in job.tech_stack[:8])
            details.append(f"🛠 <b>Tech:</b> {tech_str}")

        if job.apply_url and job.url and job.apply_url != job.url:
            details.append(f"🔗 <a href=\"{html.escape(job.apply_url)}\">Direct Apply</a>")

        body = "\n".join(details)
        return f"{header}\n{body}" if body else header

    def _format_job_markdown(self, job: Job) -> str:
        """Format a single Job object into Markdown markup."""
        title = job.title or "Untitled Position"
        company = job.company or "Unknown Company"
        url = job.apply_url or job.url

        if url:
            header = f"💼 **[{title}]({url})** @ **{company}**"
        else:
            header = f"💼 **{title}** @ **{company}**"

        details: list[str] = []

        if job.location or job.work_mode:
            loc_parts = []
            if job.location:
                loc_parts.append(f"📍 {job.location}")
            if job.work_mode:
                mode_str = job.work_mode.value if hasattr(job.work_mode, "value") else str(job.work_mode)
                loc_parts.append(f"🏢 {mode_str.capitalize()}")
            details.append(" | ".join(loc_parts))

        if job.match_score is not None:
            details.append(f"🎯 **Match Score:** {job.match_score:.1f}%")

        if job.salary_range:
            details.append(f"💰 **Salary:** {job.salary_range}")

        if job.tech_stack:
            tech_str = ", ".join(job.tech_stack[:8])
            details.append(f"🛠 **Tech:** {tech_str}")

        body = "\n".join(details)
        return f"{header}\n{body}" if body else header

    def format_alert(self, jobs: list[Job], title: Optional[str] = None) -> str:
        """Format a list of jobs into a full notification message string.

        Args:
            jobs: List of Job models.
            title: Optional header title.

        Returns:
            str: Formatted notification body.
        """
        if not jobs:
            return "No new jobs found matching your criteria."

        is_html = self.parse_mode.upper() == "HTML"
        title_str = title or f"🎯 New Job Matches ({len(jobs)})"

        if is_html:
            header = f"🔔 <b>{html.escape(title_str)}</b>\n━━━━━━━━━━━━━━━━━━━━"
            job_blocks = [self._format_job_html(job) for job in jobs]
        else:
            header = f"🔔 **{title_str}**\n━━━━━━━━━━━━━━━━━━━━"
            job_blocks = [self._format_job_markdown(job) for job in jobs]

        return f"{header}\n\n" + "\n\n".join(job_blocks)

    def split_message(self, text: str, max_len: Optional[int] = None) -> list[str]:
        """Split a formatted message into chunks within Telegram's max length limit.

        Args:
            text: Message text to split.
            max_len: Character limit per chunk (defaults to self.max_message_length).

        Returns:
            list[str]: Message chunks.
        """
        limit = max_len or self.max_message_length
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        # Split on double newline (job boundary) where possible
        sections = text.split("\n\n")
        current_chunk = ""

        for section in sections:
            if not current_chunk:
                if len(section) <= limit:
                    current_chunk = section
                else:
                    # Single section is too long; split by line or hard chunk
                    lines = section.split("\n")
                    for line in lines:
                        if len(current_chunk) + len(line) + 1 <= limit:
                            current_chunk = f"{current_chunk}\n{line}".strip()
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            # Hard split if single line exceeds limit
                            while len(line) > limit:
                                chunks.append(line[:limit])
                                line = line[limit:]
                            current_chunk = line
            else:
                candidate = f"{current_chunk}\n\n{section}"
                if len(candidate) <= limit:
                    current_chunk = candidate
                else:
                    chunks.append(current_chunk)
                    if len(section) <= limit:
                        current_chunk = section
                    else:
                        # Recurse for overlong single section
                        chunks.extend(self.split_message(section, max_len=limit))
                        current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def _send_single_chunk(
        self, client: httpx.AsyncClient, chunk: str
    ) -> bool:
        """Send a single message chunk with retry and rate-limit handling."""
        url = f"{self.api_base_url}/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": chunk,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": True,
        }

        retries = 0
        while retries <= self.max_retries:
            try:
                response = await client.post(url, json=payload, timeout=self.timeout)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        return True
                    logger.warning(f"Telegram API responded ok=false: {data}")
                    return False

                if response.status_code == 429:
                    retries += 1
                    try:
                        resp_data = response.json()
                        retry_after = resp_data.get("parameters", {}).get("retry_after", 1)
                    except Exception:
                        retry_after = int(response.headers.get("Retry-After", 1))

                    logger.warning(f"Telegram rate limited (429). Retrying after {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                # 4xx or 5xx error
                logger.error(
                    f"Telegram sendMessage failed with status {response.status_code}: {response.text}"
                )
                return False

            except (httpx.TimeoutException, httpx.RequestError) as exc:
                logger.error(f"Network error sending Telegram alert: {exc}")
                return False
            except Exception as exc:
                logger.error(f"Unexpected error sending Telegram alert: {exc}")
                return False

        return False

    async def send_alert(self, jobs: list[Job], title: Optional[str] = None) -> bool:
        """Send alert notification for a list of jobs.

        Args:
            jobs: List of Job models.
            title: Optional header title.

        Returns:
            bool: True if alert was sent successfully, False otherwise.
        """
        if not jobs:
            return True

        if not self.is_configured:
            logger.warning(
                "TelegramNotifier is not configured. Missing bot_token or chat_id."
            )
            return False

        formatted_text = self.format_alert(jobs, title=title)
        chunks = self.split_message(formatted_text)

        client = self._get_client()
        should_close = self._client is None

        try:
            for chunk in chunks:
                success = await self._send_single_chunk(client, chunk)
                if not success:
                    return False
            return True
        finally:
            if should_close:
                await client.aclose()

    async def check_health(self) -> bool:
        """Check Telegram Bot API connection and token validity.

        Returns:
            bool: True if bot token is valid and Telegram API reachable, False otherwise.
        """
        if not self.bot_token:
            return False

        url = f"{self.api_base_url}/bot{self.bot_token}/getMe"
        client = self._get_client()
        should_close = self._client is None

        try:
            response = await client.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return bool(data.get("ok", False))
            return False
        except Exception as exc:
            logger.warning(f"Telegram health check failed: {exc}")
            return False
        finally:
            if should_close:
                await client.aclose()
