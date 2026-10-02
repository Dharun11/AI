import httpx
import respx

from research_agent.fetch.fetcher import fetch_all
from research_agent.fetch.html import extract_html, extract_title
from research_agent.fetch.quality import assess

from .conftest import make_pdf


def test_html_extraction_strips_boilerplate(fixtures_dir):
    html = (fixtures_dir / "article.html").read_text(encoding="utf-8")
    text, method = extract_html(html)
    assert "under six months" in text
    assert "Section 34" in text
    assert "window.analytics" not in text
    assert "Top 10 stocks" not in text
    assert method in {"trafilatura", "bs4"}
    assert extract_title(html) == "Arbitration is faster than courts for consumer disputes"


def test_quality_flags_paywall_and_empty():
    assert assess("", 500)[0] == "parse_failed"
    assert assess("Access denied. Verify you are human.", 500)[0] == "blocked"
    long_text = "Real content sentence. " * 50
    assert assess(long_text + " Subscribe to continue reading.", 500)[0] == "partial"
    assert assess(long_text, 500) == ("ok", None)


@respx.mock
async def test_fetch_all_handles_html_pdf_and_errors(fixtures_dir):
    html = (fixtures_dir / "article.html").read_text(encoding="utf-8")
    pdf_text = "\n".join(["Consumer awards in online arbitration are fully enforceable."] * 15)
    respx.get("https://news.example/a").mock(return_value=httpx.Response(200, text=html, headers={"content-type": "text/html"}))
    respx.get("https://journal.example/paper.pdf").mock(
        return_value=httpx.Response(200, content=make_pdf(pdf_text), headers={"content-type": "application/pdf"})
    )
    respx.get("https://paywalled.example/x").mock(return_value=httpx.Response(403, text="Forbidden"))
    respx.get("https://down.example/y").mock(side_effect=httpx.ConnectError("boom"))

    sources = await fetch_all([
        "https://news.example/a",
        "https://journal.example/paper.pdf",
        "https://paywalled.example/x",
        "https://down.example/y",
    ])

    assert [s.id for s in sources] == ["S1", "S2", "S3", "S4"]
    assert "under six months" in sources[0].text
    assert sources[1].method == "pypdf"
    assert "fully enforceable" in sources[1].text
    assert sources[2].status == "blocked" and "HTTP 403: access denied to our fetcher" in sources[2].error
    assert "paywall" not in sources[2].error      # a 403 does not prove a paywall
    assert sources[3].status == "failed" and "network error" in sources[3].error


@respx.mock
async def test_a_slow_site_is_a_timeout_not_a_generic_failure():
    respx.get("https://slow.example/x").mock(side_effect=httpx.ReadTimeout("too slow"))
    (src,) = await fetch_all(["https://slow.example/x"])
    assert src.status == "timeout" and "no response within" in src.error and not src.usable


@respx.mock
async def test_unparseable_content_is_parse_failed():
    respx.get("https://bad.example/x.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4 this is not really a pdf", headers={"content-type": "application/pdf"}))
    (src,) = await fetch_all(["https://bad.example/x.pdf"])
    assert src.status == "parse_failed" and not src.usable
