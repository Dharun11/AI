"""Headless-browser rendering for JavaScript pages.

Playwright is a normal dependency, but the Chromium browser it drives is a separate one-time download:
    playwright install chromium
"""
import os
import sys
from pathlib import Path

INSTALL_HINT = "run: uv run playwright install chromium"


class BrowserUnavailable(RuntimeError):
    """Chromium is not installed, so JavaScript pages cannot be rendered."""


def _browsers_root() -> Path:
    custom = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if custom and custom != "0":
        return Path(custom)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ms-playwright"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    return Path.home() / ".cache" / "ms-playwright"


def browser_installed() -> bool:
    """Cheap check (no browser launch): is any Chromium build in Playwright's browser folder?"""
    root = _browsers_root()
    return root.is_dir() and any(root.glob("chromium*"))


async def render_html(url: str, timeout_ms: int = 10_000, settle_ms: int = 1_500) -> str:
    """Open the page and return its HTML once the document has loaded and had a moment to run its scripts.

    We wait for `domcontentloaded`, not `networkidle`: analytics, trackers and streaming requests mean many
    modern sites never go network-idle, which cost 30 seconds per page for no extra text.
    """
    if not browser_installed():
        raise BrowserUnavailable(f"Chromium is not installed ({INSTALL_HINT})")
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise BrowserUnavailable("the playwright package is not installed (run: uv sync)") from e

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_timeout(settle_ms)      # short settle so client-side rendering can fill the page
            return await page.content()
        finally:
            await browser.close()
