"""An extraction chunk that returns nothing is retried once, and if it stays empty the source says so."""
import httpx
import respx

from research_agent.config import get_settings
from research_agent.models import ClaimOut, ClaimsOut, Source
from research_agent.pipeline.extract import MIN_CHUNK_CHARS, apply_coverage, extract_claims
from research_agent.pipeline.graph import run_research
from research_agent.render.markdown import render_markdown

from .conftest import FakeLLM
from .test_api import PAGES, make_fake_llm

PARAGRAPH = "The report finds that routine jobs are shrinking as automation spreads across firms. " * 14   # ~1,150 chars


def claim(text: str) -> ClaimsOut:
    return ClaimsOut(claims=[ClaimOut(statement=f"A claim about {text}.", quote="routine jobs are shrinking as automation", facet="jobs")])


def three_part_source() -> Source:
    return Source(id="S1", url="https://a.example/1", title="Report", text="\n\n".join([PARAGRAPH] * 3))


def llm_answering(per_part: dict[str, list]):
    """Answers by 'Part i/n' in the prompt; each part has a list of answers consumed one per call."""
    calls = {}

    def answer(messages):
        body = messages[-1].content
        part = next((p for p in per_part if p in body), None)
        if part is None:                                            # chunks the test does not care about answer normally
            return claim("other")
        calls[part] = calls.get(part, 0) + 1
        answers = per_part[part]
        return answers[min(calls[part], len(answers)) - 1]

    llm = FakeLLM({"ClaimsOut": answer})
    llm.calls_by_part = calls
    return llm


def small_chunks(monkeypatch):
    monkeypatch.setattr(get_settings(), "chunk_chars", 1500)      # three paragraphs of ~1,150 chars -> several chunks


async def test_an_empty_long_chunk_is_retried_once_then_flagged(monkeypatch):
    small_chunks(monkeypatch)
    src = three_part_source()
    llm = llm_answering({"Part 1/": [claim("one")], "Part 2/": [ClaimsOut(claims=[])], "Part 3/": [claim("three")]})

    claims, coverage = await extract_claims(llm, "jobs", [src])

    total, empty = coverage["S1"]
    assert total >= 3 and empty == 1
    assert llm.calls_by_part["Part 2/"] == 2                        # the original try plus exactly one retry
    assert llm.calls_by_part["Part 1/"] == 1                        # chunks that answered are not retried
    assert len(claims) == total - 1

    (flagged,) = apply_coverage([src], coverage)
    assert flagged.status == "partial" and flagged.chunks_empty == 1
    assert f"1/{total} extraction chunks returned no claims; coverage may be incomplete" in flagged.error


async def test_a_retry_that_succeeds_leaves_the_source_unflagged(monkeypatch):
    small_chunks(monkeypatch)
    src = three_part_source()
    llm = llm_answering({"Part 1/": [claim("one")], "Part 2/": [ClaimsOut(claims=[]), claim("two")], "Part 3/": [claim("three")]})
    claims, coverage = await extract_claims(llm, "jobs", [src])
    assert coverage["S1"][1] == 0 and llm.calls_by_part["Part 2/"] == 2
    (ok,) = apply_coverage([src], coverage)
    assert ok.status == "ok" and ok.error is None


async def test_a_short_chunk_may_legitimately_be_empty(monkeypatch):
    src = Source(id="S1", url="https://a.example/1", title="Short", text="x" * (MIN_CHUNK_CHARS - 1))
    llm = llm_answering({"Part 1/1": [ClaimsOut(claims=[])]})
    claims, coverage = await extract_claims(llm, "jobs", [src])
    assert claims == [] and coverage["S1"] == (1, 0) and llm.calls_by_part["Part 1/1"] == 1   # no retry, no flag


def test_apply_coverage_keeps_an_existing_note_and_never_upgrades_a_status():
    src = Source(id="S1", url="https://a.example/1", status="partial", error="paywall detected; only the free portion was analysed")
    (out,) = apply_coverage([src], {"S1": (3, 2)})
    assert out.status == "partial"
    assert out.error.startswith("paywall detected") and "2/3 extraction chunks returned no claims" in out.error


@respx.mock
async def test_the_brief_tells_the_reader_when_part_of_a_source_was_not_analysed(monkeypatch):
    long_pages = {url: text * 3 for url, text in PAGES.items()}            # long enough that an empty answer is suspicious
    for url, text in long_pages.items():
        respx.get(url).mock(return_value=httpx.Response(
            200, text=f"<html><body><article><p>{text}</p></article></body></html>", headers={"content-type": "text/html"}))
    llm = make_fake_llm()
    normal = llm.responses["ClaimsOut"]
    c_marker = PAGES["https://c.example/3"][:40]
    llm.responses["ClaimsOut"] = lambda messages: ClaimsOut(claims=[]) if c_marker in messages[-1].content else normal(messages)

    brief = await run_research("Arbitration vs litigation in India", list(PAGES), llm=llm)

    s3 = brief.sources[2]
    assert s3.status == "partial" and s3.chunks_empty == 1 and s3.usable
    assert sum(1 for _, m in llm.calls if c_marker in m[-1].content) == 2   # S3 was asked twice, then flagged
    md = render_markdown(brief)
    assert "partial (1/1 extraction chunks returned no claims; coverage may be incomplete)" in md
    status = md.split("## Source status")[1].split("<details>")[0]
    assert status.count("processed") == 3                                   # the other three are fine
