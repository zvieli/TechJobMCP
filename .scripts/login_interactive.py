#!/usr/bin/env python3
"""Interactive browser login utility for TechJobMCP.

Launches a headful (visible) Chromium session with the persistent browser profile.
Allows the user to log into Google, Apple ID, LinkedIn, HireMeTech, or Workday,
solve 2FA / Passkeys, and save session cookies into `./browser_profile`.
The Docker container mounts this directory and inherits the authenticated state.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from playwright.async_api import async_playwright

PORTAL_URLS = {
    "google": "https://accounts.google.com",
    "apple": "https://appleid.apple.com",
    "linkedin": "https://www.linkedin.com/login",
    "hiremetech": "https://app.hireme.tech/login",
}


async def run_interactive_login(profile_dir: Path, target_urls: list[str]) -> None:
    """Launch visible browser for interactive authentication."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    abs_profile_dir = profile_dir.resolve()

    print("\n" + "=" * 70)
    print(" TechJobMCP Interactive Browser Authentication")
    print("=" * 70)
    print(f" Profile Directory: {abs_profile_dir}")
    print(f" Target Portals:    {', '.join(target_urls)}")
    print("-" * 70)
    print(" Opening visible Chromium browser...")
    print(" Please log in, complete any 2FA / SMS / Passkey verifications.")
    print(" Once logged in, return here and press ENTER to save and finish.")
    print("=" * 70 + "\n")

    launch_args = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
    ]
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(abs_profile_dir),
            headless=False,
            args=launch_args,
            user_agent=user_agent,
            viewport={"width": 1280, "height": 900},
        )

        pages = context.pages
        first_page = pages[0] if pages else await context.new_page()

        for idx, url in enumerate(target_urls):
            if idx == 0:
                page = first_page
            else:
                page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                print(f" Notice: could not auto-navigate to {url}: {e}")

        # Wait for user confirmation in terminal
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, input, ">> Press ENTER when you have finished logging in: ")

        print("\n Saving session cookies and profile state...")
        await context.close()
        print(f" Authentication state saved successfully in: {abs_profile_dir}")
        print(" TechJobMCP container will now use this authenticated session for auto-apply!\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Authenticate persistent browser session for TechJobMCP.")
    parser.add_argument(
        "--portal",
        choices=["google", "apple", "linkedin", "hiremetech", "all"],
        default="all",
        help="Target portal to log into (default: all)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom URL to open (e.g. specific Workday or career portal login page)",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=os.getenv("BROWSER_PROFILE_DIR", "./browser_profile"),
        help="Path to browser user data directory (default: ./browser_profile)",
    )

    args = parser.parse_args()

    if args.url:
        targets = [args.url]
    elif args.portal == "all":
        targets = list(PORTAL_URLS.values())
    else:
        targets = [PORTAL_URLS[args.portal]]

    profile_path = Path(args.profile_dir)
    try:
        asyncio.run(run_interactive_login(profile_path, targets))
    except KeyboardInterrupt:
        print("\n Aborted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
