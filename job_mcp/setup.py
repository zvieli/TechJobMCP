"""Interactive setup CLI for first-time browser authentication on HireMeTech."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright

from job_mcp.core.auth import (
    BASE_URL,
    DASHBOARD_PATH,
    DEFAULT_PROFILE_DIR,
    LOGIN_PATH,
)
from job_mcp.utils.logger import get_logger

logger = get_logger(__name__)


async def run_setup(profile_dir: Optional[str | Path] = None) -> bool:
    """Launch headed browser for user authentication and verify session persistence.

    Args:
        profile_dir: Directory path to save browser profile data.

    Returns:
        bool: True if authentication verification succeeded, False otherwise.
    """
    if profile_dir is None:
        profile_dir = os.getenv("BROWSER_PROFILE_DIR", DEFAULT_PROFILE_DIR)

    profile_path = Path(profile_dir).expanduser().resolve()
    profile_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("       HireMeTech MCP Server - First-Time Authentication")
    print("=" * 65)
    print(f"Browser Profile Directory: {profile_path}")
    print("\nA Chromium browser window will now open.")
    print("Please follow these steps:")
    print("  1. Log in to your HireMeTech account in the opened browser window.")
    print("  2. Complete any 2-factor authentication or SSO if required.")
    print("  3. Ensure you are on the HireMeTech Dashboard page.")
    print("  4. Return to this terminal and press [Enter].")
    print("=" * 65)

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

    pw = await async_playwright().start()
    try:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_path),
            headless=False,
            args=launch_args,
            user_agent=user_agent,
            viewport=viewport,
        )

        page = context.pages[0] if context.pages else await context.new_page()

        login_url = os.getenv("HIREMETECH_LOGIN_URL", f"{BASE_URL}")
        print(f"\nNavigating to {login_url}...")
        try:
            await page.goto(login_url, wait_until="commit", timeout=20000)
            if hasattr(page, "wait_for_timeout") and callable(page.wait_for_timeout):
                await page.wait_for_timeout(3000)

            # Check if there is an explicit login button to click
            if hasattr(page, "locator"):
                loc = page.locator("button:has-text('התחברות'), button:has-text('Login'), button:has-text('Sign in'), a[href*='login']")
                if hasattr(loc, "count"):
                    count = await loc.count() if asyncio.iscoroutinefunction(loc.count) or asyncio.iscoroutine(loc.count()) else (loc.count() if callable(loc.count) else 0)
                    if count > 0 and hasattr(loc, "first"):
                        try:
                            await loc.first.click(timeout=3000)
                            print("Opened login dialog/form.")
                        except Exception:
                            pass
        except Exception as exc:
            print(f"Notice: Initial page load encountered: {exc}")

        # Wait for user to interact and press Enter in the terminal
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, input, "\n👉 Please complete your login in the browser window.\n👉 Once logged in and viewing your jobs/dashboard, press [ENTER] here: ")

        print("\nVerifying session authentication status...")
        dashboard_url = os.getenv("HIREMETECH_DASHBOARD_URL", f"{BASE_URL}{DASHBOARD_PATH}")
        try:
            response = await page.goto(dashboard_url, wait_until="commit", timeout=20000)
            if hasattr(page, "wait_for_timeout") and callable(page.wait_for_timeout):
                await page.wait_for_timeout(3000)
            status_code = response.status if response and hasattr(response, "status") else 200
        except Exception as exc:
            print(f"Warning during verification navigation: {exc}")
            status_code = 0

        current_url = page.url or ""
        is_authenticated = (
            LOGIN_PATH not in current_url
            and status_code not in (401, 403)
        )

        if is_authenticated:
            print("\n" + "=" * 65)
            print("  [SUCCESS] Authentication Verified Successfully!")
            print(f"  Session cookies & state saved to: {profile_path}")
            print("  You can now launch the MCP server in headless mode:")
            print("    python -m hireme_mcp")
            print("=" * 65 + "\n")
            await context.close()
            return True
        else:
            print("\n" + "=" * 65)
            print("  [FAILED] Authentication verification failed.")
            print(f"  Current browser URL: {current_url}")
            print("  The session appears to not be logged in.")
            print("  Please re-run this setup and complete login before pressing Enter.")
            print("=" * 65 + "\n")
            await context.close()
            return False

    finally:
        await pw.stop()


def main() -> None:
    """CLI entry point for hireme-mcp-setup."""
    success = asyncio.run(run_setup())
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
