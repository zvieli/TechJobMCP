"""Authentication and browser session management for HireMeTech MCP server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from hireme_mcp.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
BASE_URL = os.getenv("HIREMETECH_BASE_URL", "https://hiremetech.com")
DASHBOARD_PATH = os.getenv("HIREMETECH_DASHBOARD_PATH", "/he-il/jobs-app")
LOGIN_PATH = os.getenv("HIREMETECH_LOGIN_PATH", "/login")
DEFAULT_PROFILE_DIR = os.path.expanduser("~/.hireme_mcp/browser_profile")


class SessionManager:
    """Manages persistent browser session and authentication state for HireMeTech."""

    def __init__(
        self,
        user_data_dir: Optional[str | Path] = None,
        headless: Optional[bool] = None,
    ) -> None:
        """Initialize SessionManager with profile directory and headless mode settings."""
        if user_data_dir is None:
            user_data_dir = os.getenv("BROWSER_PROFILE_DIR", DEFAULT_PROFILE_DIR)

        self.user_data_dir: Path = Path(user_data_dir).expanduser().resolve()

        if headless is None:
            env_headless = os.getenv("BROWSER_HEADLESS", "true").strip().lower()
            self.headless: bool = env_headless in ("true", "1", "yes")
        else:
            self.headless = headless

        self.playwright: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._initialized: bool = False

    async def initialize(self) -> None:
        """Start Playwright and launch Chromium persistent context."""
        if self._initialized and self.context is not None:
            logger.debug("SessionManager is already initialized.")
            return

        logger.info(
            "Initializing SessionManager (profile_dir=%s, headless=%s)",
            self.user_data_dir,
            self.headless,
        )
        self.user_data_dir.mkdir(parents=True, exist_ok=True)

        self.playwright = await async_playwright().start()

        launch_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ]
        user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        viewport = {"width": 1280, "height": 800}

        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.user_data_dir),
            headless=self.headless,
            args=launch_args,
            user_agent=user_agent,
            viewport=viewport,
        )

        if self.context.pages:
            self.page = self.context.pages[0]
        else:
            self.page = await self.context.new_page()

        self._initialized = True
        logger.info("SessionManager initialized successfully.")

    async def get_page(self) -> Page:
        """Get or create the active browser page."""
        if not self._initialized or self.context is None:
            await self.initialize()

        if self.page is None or self.page.is_closed():
            if self.context and self.context.pages:
                # Find an open page or create a new one
                open_pages = [p for p in self.context.pages if not p.is_closed()]
                if open_pages:
                    self.page = open_pages[0]
                else:
                    self.page = await self.context.new_page()
            elif self.context:
                self.page = await self.context.new_page()
            else:
                raise RuntimeError("Browser context is unavailable.")

        return self.page

    async def check_session_health(self) -> bool:
        """Check if session is authenticated, attempting fast /api/auth/me probe first.

        Returns:
            bool: True if session is valid, False if unauthenticated or redirected to login.
        """
        page = await self.get_page()

        # 1. Fast API health check probe (<50ms)
        try:
            auth_me_url = f"{BASE_URL}/api/auth/me"
            resp = await page.request.get(auth_me_url, timeout=5000)
            if resp.status == 200:
                data = await resp.json()
                if isinstance(data, dict) and "user" in data and isinstance(data["user"], dict) and data["user"].get("id"):
                    user = data["user"]
                    user_label = user.get("email") or user.get("name") or user.get("id")
                    logger.info(
                        "Session health check passed via /api/auth/me for user %s",
                        user_label,
                    )
                    return True
            elif resp.status in (401, 403):
                logger.warning(
                    "Session unauthenticated (HTTP %d on /api/auth/me)",
                    resp.status,
                )
                return False
        except Exception as exc:
            logger.debug(
                "Fast /api/auth/me check failed with error: %s. Falling back to page navigation.",
                exc,
            )

        # 2. Fallback: Standard page navigation check
        target_url = f"{BASE_URL}{DASHBOARD_PATH}"
        logger.info("Checking session health at %s", target_url)

        try:
            response = await page.goto(
                target_url,
                wait_until="commit",
                timeout=10000,
            )
            await page.wait_for_timeout(2500)

            if response and response.status in (401, 403):
                logger.warning(
                    "Session health check failed: received HTTP %d status code.",
                    response.status,
                )
                return False

            current_url = page.url
            if LOGIN_PATH in current_url:
                logger.warning(
                    "Session health check failed: redirected to login URL (%s)",
                    current_url,
                )
                return False

            logger.info("Session health check passed for %s", current_url)
            return True

        except Exception as exc:
            logger.warning("Session health check encountered error: %s", exc)
            return False

    async def attempt_reauth(self) -> bool:
        """Re-attempt page navigation/refresh and verify session health.

        Returns:
            bool: True if session is healthy after retry, False otherwise.
        """
        logger.info("Attempting session re-authentication/refresh...")
        try:
            page = await self.get_page()
            await page.reload(wait_until="commit", timeout=20000)
            await page.wait_for_timeout(2500)
        except Exception as exc:
            logger.warning("Reload failed during attempt_reauth: %s", exc)

        return await self.check_session_health()

    async def recover(self) -> bool:
        """Tear down browser state and re-initialize from scratch.

        Returns:
            bool: True if session is healthy after recovery, False otherwise.
        """
        logger.warning("Starting full session recovery...")
        try:
            await self.shutdown()
        except Exception as exc:
            logger.debug("Error during recovery shutdown: %s", exc)

        self._initialized = False

        try:
            await self.initialize()
        except Exception as exc:
            logger.error("Recovery re-initialization failed: %s", exc)
            return False

        healthy = await self.check_session_health()
        if healthy:
            logger.info("Session recovery succeeded.")
        else:
            logger.warning("Session recovery completed but health check failed.")
        return healthy

    async def ensure_ready(self, max_retries: int = 3) -> Page:
        """Initialize, health-check, and auto-recover the browser session.

        Retries up to max_retries times, performing full recovery between attempts.

        Args:
            max_retries: Maximum number of initialization/recovery attempts.

        Returns:
            Page: A healthy, ready-to-use Playwright Page.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                if not self._initialized:
                    await self.initialize()

                if await self.check_session_health():
                    return await self.get_page()

                # Unhealthy — try recovery
                logger.warning(
                    "Session unhealthy on attempt %d/%d, recovering...",
                    attempt,
                    max_retries,
                )
                if await self.recover():
                    return await self.get_page()

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Session attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                try:
                    await self.shutdown()
                except Exception:
                    pass
                # Reset state for next attempt
                self._initialized = False

        raise RuntimeError(
            f"Browser session failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def inject_session_storage(self, state: dict[str, Any]) -> None:
        """Set sessionStorage items (e.g. jobBoardState) in the active page.

        Args:
            state: Dictionary of key-value pairs to set in sessionStorage.
        """
        page = await self.get_page()
        logger.info("Injecting %d items into sessionStorage", len(state))
        await page.evaluate(
            """(data) => {
                for (const [key, value] of Object.entries(data)) {
                    const valStr = typeof value === 'string' ? value : JSON.stringify(value);
                    sessionStorage.setItem(key, valStr);
                }
            }""",
            state,
        )

    async def shutdown(self) -> None:
        """Cleanly close page, browser context, and stop Playwright."""
        logger.info("Shutting down SessionManager...")
        if self.page and not self.page.is_closed():
            try:
                await self.page.close()
            except Exception as exc:
                logger.debug("Error closing page: %s", exc)
        self.page = None

        if self.context:
            try:
                await self.context.close()
            except Exception as exc:
                logger.debug("Error closing context: %s", exc)
        self.context = None

        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as exc:
                logger.debug("Error stopping Playwright: %s", exc)
        self.playwright = None

        self._initialized = False
        logger.info("SessionManager shutdown complete.")
