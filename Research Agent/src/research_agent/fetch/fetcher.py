"""Fetch a URL and turn it into a `Source` with clean text. Never raises: failures become status=failed."""
import asyncio
import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..models import Source
from .html import extract_html, extract_title
from .js import render_html
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


async def fetch_source(source_id: str, url: str, client: httpx.AsyncClient) -> Source:
    s = get_settings()
    src = Source(id=source_id, url=url)
    try:
        resp = await _get(client, url)
    except Exception as e:
        log.warning("fetch failed for %s: %s", url, e)
        src.status, src.error = "failed", f"network error: {type(e).__name__}"
        return src

    if resp.status_code in (401, 402, 403, 451) and not s.use_playwright:
        src.status, src.error = "failed", f"HTTP {resp.status_code} (paywall or access restriction)"
        return src
    if resp.status_code >= 400 and resp.status_code not in (401, 402, 403, 451):
        src.status, src.error = "failed", f"HTTP {resp.status_code}"
        return src

    try:
        if _is_pdf(resp, url):
            text, title = await asyncio.to_thread(extract_pdf, resp.content)
            src.title, src.method = title, "pypdf"
        else:
            html = resp.text
            text, src.method = await asyncio.to_thread(extract_html, html)
            src.title = extract_title(html)
            if (len(text) < s.min_text_chars or resp.status_code >= 400) and s.use_playwright:
                log.info("re-rendering %s with Playwright (%d chars static)", url, len(text))
                rendered = await render_html(url, int(s.fetch_timeout * 1000))
                js_text, _ = await asyncio.to_thread(extract_html, rendered)
                if len(js_text) > len(text):
                    text, src.method = js_text, "playwright"
                    src.title = extract_title(rendered) or src.title
    except Exception as e:
        log.warning("parse failed for %s: %s", url, e)
        src.status, src.error = "failed", f"parse error: {type(e).__name__}: {e}"
        return src

    src.status, src.error = assess(text, s.min_text_chars)
    if len(text) > s.max_chars_per_source:
        text = text[: s.max_chars_per_source]
        src.error = (src.error + "; " if src.error else "") + f"truncated to {s.max_chars_per_source} chars"
    src.text = text
    src.title = src.title or url
    return src


async def fetch_all(urls: list[str]) -> list[Source]:
    s = get_settings()
    async with httpx.AsyncClient(headers=HEADERS, timeout=s.fetch_timeout, follow_redirects=True) as client:
        return list(await asyncio.gather(*(fetch_source(f"S{i}", u, client) for i, u in enumerate(urls, 1))))
