"""API validation + an end-to-end run of the graph with mocked HTTP and a fake LLM."""
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from research_agent import api
from research_agent.config import get_settings
from research_agent.models import (ClaimOut, ClaimsOut, CoverageOut, FacetCoverage, FacetsOut, Gap, GroupOut,
                                   GroupsOut, TldrOut)
from research_agent.pipeline import graph as graph_mod

from .conftest import FakeLLM

client = TestClient(api.app)

PAGES = {
    "https://a.example/1": "Online arbitration of consumer disputes takes on average under six months. " * 12,
    "https://b.example/2": "Arbitration usually concludes in under six months, far faster than litigation. " * 12,
    "https://c.example/3": "AI-driven case allocation reduces arbitrator bias by fifteen percent in our trials. " * 12,
    "https://d.example/4": "Section 34 appeals frequently stay arbitration awards for up to eighteen months. " * 12,
}

CLAIMS_BY_URL = {
    "https://a.example/1": ClaimOut(statement="Online arbitration takes under six months on average.",
                          quote="takes on average under six months", facet="timeline"),
    "https://b.example/2": ClaimOut(statement="Arbitration concludes in under six months.",
                          quote="usually concludes in under six months", facet="timeline"),
    "https://c.example/3": ClaimOut(statement="AI case allocation reduces bias by 15%.",
                          quote="reduces arbitrator bias by fifteen percent", facet="fairness"),
    "https://d.example/4": ClaimOut(statement="Section 34 appeals stay awards for up to 18 months.",
                          quote="stay arbitration awards for up to eighteen months", facet="enforceability"),
}


def _extract(messages):
    body = messages[-1].content
    out = [CLAIMS_BY_URL[url] for url, text in PAGES.items() if text[:40] in body]
    # plus a hallucinated claim that verify must reject
    out.append(ClaimOut(statement="Made up.", quote="this sentence never appears anywhere in the text", facet="x"))
    return ClaimsOut(claims=out)


def make_fake_llm():
    return FakeLLM({
        "ClaimsOut": _extract,
        "GroupsOut": GroupsOut(groups=[
            GroupOut(kind="consensus", summary="Arbitration resolves disputes in under six months.",
                     claim_ids=["S1-C01", "S2-C01"]),
            GroupOut(kind="contradiction", summary="Are awards quickly enforceable?",
                     claim_ids=["S1-C01", "S4-C01"], positions=[["S1-C01"], ["S4-C01"]],
                     position_labels=["Arbitration is fast", "Appeals can stall awards for months"]),
            GroupOut(kind="consensus", summary="bogus", claim_ids=["S3-C01", "S3-C09"]),
        ]),
        "FacetsOut": FacetsOut(facets=[
            Gap(facet="Timelines", why_it_matters="speed"),
            Gap(facet="Data privacy (DPDP Act)", why_it_matters="compliance"),
            Gap(facet="Vernacular language support", why_it_matters="access"),
        ]),
        # "Data privacy" claimed covered but cites no real claim -> must still be reported as a gap
        "CoverageOut": CoverageOut(covered=[FacetCoverage(facet="Timelines", claim_ids=["S1-C01"]),
                                            FacetCoverage(facet="Data privacy (DPDP Act)", claim_ids=["S9-C99"])]),
        "TldrOut": TldrOut(bullets=["one", "two", "three"]),
    })


@pytest.mark.parametrize("n", [2, 6])
def test_rejects_wrong_url_count(n):
    r = client.post("/research", json={"topic": "x topic", "urls": [f"https://e{i}.example/" for i in range(n)]})
    assert r.status_code == 422


def test_rejects_duplicate_urls():
    r = client.post("/research", json={"topic": "x topic", "urls": ["https://a.example/", "https://a.example", "https://b.example/"]})
    assert r.status_code == 422


def section(md: str, name: str) -> str:
    """Text of the '## name' section, up to the next '## ' heading (sub-headings use '###')."""
    marker = "\n## "
    return md.split(marker + name + "\n")[1].split(marker)[0]


@respx.mock
def test_end_to_end_brief(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "output_dir", tmp_path)
    for url, text in PAGES.items():
        respx.get(url).mock(return_value=httpx.Response(200, text=f"<html><body><article><p>{text}</p></article></body></html>",
                                                       headers={"content-type": "text/html"}))
    fake = make_fake_llm()
    monkeypatch.setattr(graph_mod, "get_llm", lambda: fake)

    r = client.post("/research", json={"topic": "Arbitration vs litigation in India", "urls": list(PAGES)})
    assert r.status_code == 200, r.text
    md = r.text

    # consensus cites two distinct sources with real URLs
    consensus = section(md, "Consensus")
    assert "https://a.example/1" in consensus and "https://b.example/2" in consensus
    assert "Sources: 2 of 4 · Evidence: S1-C01, S2-C01" in consensus        # evidence indicator
    # contradiction: explicit positions with their own stance, sources and a "vs." between them
    contra = section(md, "Contradictions")
    assert "**Position A: Arbitration is fast**" in contra and "**Position B: Appeals can stall awards for months**" in contra
    assert "Sources: S1 · Evidence: S1-C01" in contra and "Sources: S4 · Evidence: S4-C01" in contra
    assert "**vs.**" in contra and "https://d.example/4" in contra
    # bogus single-source consensus became a single-source claim
    assert "AI case allocation reduces bias" in section(md, "Single-source claims")
    # gaps are uncovered expected facets
    gaps = section(md, "Gaps")
    assert "Data privacy (DPDP Act)" in gaps and "Timelines" not in gaps
    # every source is listed with its status
    status = section(md, "Source status")
    assert status.count("processed") == 4 and "✓" in status
    # hallucinated claims never appear; rejection counted
    assert "Made up." not in md
    assert "4 rejected as ungrounded" in md
    # evidence chain: claim ID -> source -> exact quote -> URL
    chain = md.split("## Evidence chain")[1]
    assert '| `S2-C01` | S2 b.example | "usually concludes in under six months" | https://b.example/2 |' in chain
    assert list(tmp_path.glob("*.md"))


@respx.mock
def test_502_when_too_few_sources(monkeypatch):
    for i, url in enumerate(PAGES):
        respx.get(url).mock(return_value=httpx.Response(404 if i else 200, text="<p>" + "x " * 400 + "</p>"))
    monkeypatch.setattr(graph_mod, "get_llm", make_fake_llm)
    r = client.post("/research", json={"topic": "Arbitration", "urls": list(PAGES)[:3]})
    assert r.status_code == 502
