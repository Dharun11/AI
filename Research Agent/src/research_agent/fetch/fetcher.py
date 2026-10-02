"""Fetch a URL and turn it into a `Source` with clean text. Never raises: failures become status=failed."""
import asyncio
import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..models import Source
from .html import extract_html, extract_title
from .js import INSTALL_HINT, BrowserUnavailable, render_html
from .pdf import extract_pdf
from .quality import assess

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    reraise=True,
)
async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    return await client.get(url)


def _is_pdf(resp: httpx.Response, url: str) -> bool:
    ctype = resp.headers.get("content-type", "").lower()
    return "application/pdf" in ctype or url.lower().split("?")[0].endswith(".pdf") or resp.content[:5] == b"%PDF-"


BLOCKED_CODES = (401, 402, 403, 451)    # our fetcher was refused; a real browser may still be let in


def _browser_failure_note(e: Exception) -> str:
    if type(e).__name__ == "TimeoutError":          # Playwright's TimeoutError
        return f"browser retry timed out after {get_settings().browser_timeout:g}s"
    return f"browser rendering failed: {type(e).__name__}"


async def _render_with_browser(url: str, static_text: str, src: Source) -> tuple[str, str | None]:
    """Re-render a thin or blocked page in Chromium. On any failure keep the static text and say why."""
    s = get_settings()
    log.info("re-rendering %s in the browser (%d chars from plain download)", url, len(static_text))
    try:
        rendered = await render_html(url, int(s.browser_timeout * 1000), int(s.browser_settle * 1000))
    except BrowserUnavailable:
        return static_text, f"could not retry in a browser: Chromium not installed ({INSTALL_HINT})"
    except Exception as e:
        log.warning("browser rendering failed for %s: %s", url, e)
        return static_text, _browser_failure_note(e)
    js_text, _ = await asyncio.to_thread(extract_html, rendered)
    if len(js_text) > len(static_text):
        src.method = "playwright"
        src.title = extract_title(rendered) or src.title
        return js_text, None
    return static_text, None


def _join(*notes: str | None) -> str | None:
    return "; ".join(n for n in notes if n) or None


async def fetch_source(source_id: str, url: str, client: httpx.AsyncClient) -> Source:
    s = get_settings()
    src = Source(id=source_id, url=url)
    try:
        resp = await _get(client, url)
    except httpx.TimeoutException:
        log.warning("fetch timed out for %s", url)
        src.status, src.error = "timeout", f"no response within {s.fetch_timeout:g}s"
        return src
    except Exception as e:
        log.warning("fetch failed for %s: %s", url, e)
        src.status, src.error = "failed", f"network error: {type(e).__name__}"
        return src

    # A refusal proves our fetcher was denied, not that the page is paywalled, so the wording stays neutral.
    blocked = resp.status_code in BLOCKED_CODES
    reason = f"HTTP {resp.status_code}: access denied to our fetcher" if blocked else f"HTTP {resp.status_code}"
    if resp.status_code >= 400 and not (blocked and s.use_playwright):
        src.status, src.error = ("blocked" if blocked else "failed"), reason
        return src

    browser_note = None
    try:
        if _is_pdf(resp, url):
            text, title = await asyncio.to_thread(extract_pdf, resp.content)
            src.title, src.method = title, "pypdf"
        else:
            html = resp.text
            text, src.method = await asyncio.to_thread(extract_html, html)
            src.title = extract_title(html)
            if (len(text) < s.min_text_chars or resp.status_code >= 400) and s.use_playwright:
                text, browser_note = await _render_with_browser(url, text, src)
    except Exception as e:
        log.warning("parse failed for %s: %s", url, e)
        src.status, src.error = "parse_failed", f"{type(e).__name__}: {e}"
        return src

    if resp.status_code >= 400 and src.method != "playwright":
        # The plain download was refused and the browser did not get through, so the text is an error page.
        src.status, src.error = "blocked", _join(reason, browser_note)
        return src

    src.status, reason_ = assess(text, s.min_text_chars)
    src.error = _join(reason_, browser_note)
    if len(text) > s.max_chars_per_source:
        text = text[: s.max_chars_per_source]
        src.error = _join(src.error, f"truncated to {s.max_chars_per_source} chars")
    src.text = text
    src.title = src.title or url
    return src


async def fetch_all(urls: list[str]) -> list[Source]:
    s = get_settings()
    async with httpx.AsyncClient(headers=HEADERS, timeout=s.fetch_timeout, follow_redirects=True) as client:
        return list(await asyncio.gather(*(fetch_source(f"S{i}", u, client) for i, u in enumerate(urls, 1))))
