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
    assert assess("", 500)[0] == "failed"
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
    assert sources[2].status == "failed" and "403" in sources[2].error
    assert sources[3].status == "failed" and "network error" in sources[3].error
