"""Browser (Playwright) fallback: on by default, and a missing or failing browser never loses the static text."""
import httpx
import pytest
import respx

from research_agent.config import Settings, get_settings
from research_agent.fetch import fetcher
from research_agent.fetch.fetcher import fetch_all
from research_agent.fetch.js import BrowserUnavailable, browser_installed

URL = "https://spa.example/page"
THIN = "<html><head><title>App shell</title></head><body><div id='root'></div><p>Loading...</p></body></html>"
RICH = ("<html><head><title>Rendered article</title></head><body><article><p>"
        + "A real sentence from the rendered article about jobs. " * 30 + "</p></article></body></html>")


def serve(status: int, html: str):
    respx.get(URL).mock(return_value=httpx.Response(status, text=html, headers={"content-type": "text/html"}))


def render_raises(exc: Exception):
    async def render(url, timeout_ms=0, settle_ms=0):
        raise exc
    return render


def render_returns(html: str):
    async def render(url, timeout_ms=0, settle_ms=0):
        return html
    return render


def test_browser_rendering_is_on_by_default(monkeypatch):
    monkeypatch.delenv("USE_PLAYWRIGHT", raising=False)       # ignore whatever the developer's .env says
    assert Settings(_env_file=None).use_playwright is True


def test_browser_installed_looks_in_playwright_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
    assert not browser_installed()
    (tmp_path / "chromium_headless_shell-1234").mkdir()
    assert browser_installed()


@respx.mock
async def test_thin_page_renders_in_the_browser_and_says_so(monkeypatch):
    serve(200, THIN)
    monkeypatch.setattr(fetcher, "render_html", render_returns(RICH))
    (src,) = await fetch_all([URL])
    assert src.status == "ok" and src.method == "playwright"
    assert "real sentence from the rendered article" in src.text and src.title == "Rendered article"


@respx.mock
async def test_missing_browser_keeps_static_text_and_explains_how_to_fix_it(monkeypatch):
    serve(200, THIN)
    monkeypatch.setattr(fetcher, "render_html", render_raises(BrowserUnavailable("no chromium")))
    (src,) = await fetch_all([URL])
    assert src.status == "partial" and src.method != "playwright"      # not "failed": we still have the static text
    assert "Loading" in src.text
    assert "Chromium not installed" in src.error and "playwright install chromium" in src.error


@respx.mock
async def test_a_crashing_browser_degrades_to_partial_not_failed(monkeypatch):
    serve(200, THIN)
    monkeypatch.setattr(fetcher, "render_html", render_raises(RuntimeError("boom")))
    (src,) = await fetch_all([URL])
    assert src.status == "partial" and "browser rendering failed: RuntimeError" in src.error


@respx.mock
async def test_blocked_page_is_retried_in_the_browser(monkeypatch):
    serve(403, "Forbidden")
    monkeypatch.setattr(fetcher, "render_html", render_returns(RICH))
    (src,) = await fetch_all([URL])
    assert src.status == "ok" and src.method == "playwright"


@respx.mock
async def test_blocked_page_with_no_browser_is_failed_with_the_http_code(monkeypatch):
    serve(403, "Forbidden")
    monkeypatch.setattr(fetcher, "render_html", render_raises(BrowserUnavailable("no chromium")))
    (src,) = await fetch_all([URL])
    assert src.status == "blocked" and src.text == ""
    assert "HTTP 403" in src.error and "Chromium not installed" in src.error


@respx.mock
async def test_browser_is_not_used_when_the_page_is_fine_or_the_setting_is_off(monkeypatch):
    def forbidden(url, timeout_ms=0, settle_ms=0):
        raise AssertionError("the browser must not be launched here")
    monkeypatch.setattr(fetcher, "render_html", forbidden)

    serve(200, RICH)                                   # plenty of text: no browser needed
    (src,) = await fetch_all([URL])
    assert src.status == "ok" and src.method != "playwright"

    monkeypatch.setattr(get_settings(), "use_playwright", False)
    serve(200, THIN)                                   # thin, but the user turned the fallback off
    (src,) = await fetch_all([URL])
    assert src.status == "partial"

    serve(403, "Forbidden")
    (src,) = await fetch_all([URL])
    assert src.status == "blocked" and "access denied to our fetcher" in src.error


@pytest.mark.parametrize("code", [404, 500])
@respx.mock
async def test_other_http_errors_fail_without_trying_the_browser(monkeypatch, code):
    serve(code, "nope")
    monkeypatch.setattr(fetcher, "render_html", render_raises(AssertionError("browser must not be launched")))
    (src,) = await fetch_all([URL])
    assert src.status == "failed" and f"HTTP {code}" in src.error and "access denied" not in src.error


@respx.mock
async def test_a_browser_timeout_is_reported_as_a_timeout_note(monkeypatch):
    serve(200, THIN)
    monkeypatch.setattr(fetcher, "render_html", render_raises(TimeoutError("Page.goto: Timeout 10000ms exceeded")))
    (src,) = await fetch_all([URL])
    assert src.status == "partial" and "browser retry timed out after 10s" in src.error


@respx.mock
async def test_blocked_page_whose_browser_retry_times_out_stays_blocked_with_both_reasons(monkeypatch):
    serve(403, "Forbidden")
    monkeypatch.setattr(fetcher, "render_html", render_raises(TimeoutError("slow")))
    (src,) = await fetch_all([URL])
    assert src.status == "blocked" and "HTTP 403" in src.error and "timed out" in src.error


@respx.mock
async def test_the_browser_is_given_the_short_timeout_and_settle_from_settings(monkeypatch):
    seen = {}

    async def render(url, timeout_ms=0, settle_ms=0):
        seen.update(timeout_ms=timeout_ms, settle_ms=settle_ms)
        return RICH

    serve(200, THIN)
    monkeypatch.setattr(fetcher, "render_html", render)
    await fetch_all([URL])
    assert seen == {"timeout_ms": 10_000, "settle_ms": 1_500}      # domcontentloaded budget, not 30s of network-idle
