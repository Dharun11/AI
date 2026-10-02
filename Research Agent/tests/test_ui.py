"""Drive the Streamlit app headlessly with mocked HTTP and a fake LLM."""
from pathlib import Path

import httpx
import respx
from streamlit.testing.v1 import AppTest

from research_agent.pipeline import graph as graph_mod

from .test_api import PAGES, make_fake_llm

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def test_parse_urls_handles_separators_duplicates_and_junk():
    # parse_urls is pure, so exec just its source instead of re-running the whole page
    src = Path(APP).read_text(encoding="utf-8")
    ns: dict = {}
    exec("import re\nfrom urllib.parse import urlparse\n" + src[src.index("def parse_urls"):src.index("def esc")], ns)
    urls, errors = ns["parse_urls"]("https://a.example/1, https://b.example/2\nhttps://a.example/1/ ftp://x nope")
    assert urls == ["https://a.example/1", "https://b.example/2"]
    assert len(errors) == 3


def test_validation_errors_are_shown_and_nothing_runs():
    at = _app()
    at.text_input(key="topic").set_value("AI and jobs")
    at.text_area(key="urls_text").set_value("https://a.example/1\nhttps://b.example/2")
    at.button[1].click().run()  # button[0] is "Load example"
    assert any("3 to 5 unique URLs" in e.value for e in at.error)
    assert "result" not in at.session_state


@respx.mock
def test_full_run_renders_every_section(monkeypatch):
    for url, text in PAGES.items():
        respx.get(url).mock(return_value=httpx.Response(
            200, text=f"<html><body><article><p>{text}</p></article></body></html>", headers={"content-type": "text/html"}))
    monkeypatch.setattr(graph_mod, "get_llm", make_fake_llm)

    at = _app()
    at.text_input(key="topic").set_value("Arbitration vs litigation in India")
    at.text_area(key="urls_text").set_value("\n".join(PAGES))
    at.button[1].click().run()

    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert "result" in at.session_state
    headers = [h.value for h in at.subheader]
    assert any(h.startswith("Consensus (1)") for h in headers)
    assert any(h.startswith("Contradictions (1)") for h in headers)
    assert any(h.startswith("Single-source claims") for h in headers)
    assert any(h.startswith("Gaps (2)") for h in headers)
    body = " ".join(m.value for m in at.markdown)
    assert "https://a.example/1" in body and "https://d.example/4" in body
    assert "Position A: Arbitration is fast" in body and "Position B: Appeals can stall awards for months" in body
    assert "Sources: 2 of 4" in " ".join(c.value for c in at.caption)
    assert any("Source status" in t.label for t in at.tabs)


def test_sidebar_tells_the_user_whether_browser_rendering_works(monkeypatch):
    monkeypatch.setattr("research_agent.fetch.js.browser_installed", lambda: False)
    at = _app()
    warnings = " ".join(w.value for w in at.sidebar.warning)
    assert "Chromium is not installed" in warnings and "playwright install chromium" in warnings

    monkeypatch.setattr("research_agent.fetch.js.browser_installed", lambda: True)
    at = _app()
    assert "Chromium ready" in " ".join(m.value for m in at.sidebar.markdown)



@respx.mock
def test_source_strip_shows_partial_and_failed_sources(monkeypatch):
    pages = list(PAGES)
    respx.get(pages[0]).mock(return_value=httpx.Response(200, text="<html><body><article><p>" + PAGES[pages[0]] + "</p></article></body></html>"))
    respx.get(pages[1]).mock(return_value=httpx.Response(200, text="<html><body><article><p>" + PAGES[pages[1]] + "</p></article></body></html>"))
    respx.get(pages[2]).mock(return_value=httpx.Response(200, text="<html><body><p>Loading...</p></body></html>"))   # thin page
    respx.get(pages[3]).mock(return_value=httpx.Response(404, text="gone"))
    monkeypatch.setattr(graph_mod, "get_llm", make_fake_llm)

    at = _app()
    at.text_input(key="topic").set_value("Arbitration vs litigation in India")
    at.text_area(key="urls_text").set_value("\n".join(pages))
    at.button[1].click().run()

    assert not at.exception, [e.value for e in at.exception]
    strip = next(m.value for m in at.markdown if "c.example" in m.value and "d.example" in m.value)
    assert ":green[✓ **a.example**] processed" in strip
    assert ":orange[⚠ **c.example**] partial" in strip and "Chromium is not available" not in strip   # thin page, no browser
    assert ":red[✗ **d.example**] failed (HTTP 404)" in strip
