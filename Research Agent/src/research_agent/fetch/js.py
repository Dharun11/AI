"""Headless-browser rendering for JavaScript-heavy pages (optional dependency)."""


async def render_html(url: str, timeout_ms: int = 30_000) -> str:
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:  # pragma: no cover - optional extra
        raise RuntimeError("Playwright not installed: pip install -e '.[js]' && playwright install chromium") from e

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            return await page.content()
        finally:
            await browser.close()
