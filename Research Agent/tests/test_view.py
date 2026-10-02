from research_agent.models import Brief, Claim, ClaimGroup, Source
from research_agent.render.view import evidence_rows, indicator, position_indicator, short_name, source_rows, status_label


def _brief():
    sources = [
        Source(id="S1", url="https://www.example.com/a", title="A", status="ok", method="trafilatura"),
        Source(id="S2", url="https://b.example/x", title="B", status="ok", method="playwright"),
        Source(id="S3", url="https://c.example/y", title="C", status="partial", error="paywall detected"),
        Source(id="S4", url="https://d.example/z", title="D", status="blocked", error="HTTP 403: access denied to our fetcher"),
        Source(id="S5", url="https://e.example/z", title="E", status="timeout", error="no response within 30s"),
        Source(id="S6", url="https://f.example/z", title="F", status="failed", error="HTTP 404"),
    ]
    claims = {cid: Claim(id=cid, source_id=cid[:2], statement=f"stmt {cid}", quote=f"quote  for\n{cid}", facet="f")
              for cid in ["S1-C01", "S2-C01", "S3-C01"]}
    return Brief(topic="t", sources=sources, claims=claims)


def test_short_name_drops_www():
    assert short_name(_brief().sources[0]) == "example.com"


def test_status_labels_say_what_happened():
    s = _brief().sources
    assert status_label(s[0]) == "processed"
    assert status_label(s[1]) == "processed (browser-rendered)"       # the user can see Playwright was used
    assert status_label(s[2]) == "partial (paywall detected)"
    assert status_label(s[3]) == "inaccessible (HTTP 403: access denied to our fetcher)"      # not called a paywall
    assert status_label(s[4]) == "timed out (no response within 30s)"
    assert status_label(s[5]) == "failed (HTTP 404)"
    assert [r["icon"] for r in source_rows(_brief())] == ["✓", "✓", "⚠", "⚠", "⚠", "✗"]


def test_indicators_count_distinct_sources_out_of_those_read():
    b = _brief()
    assert indicator(b, ["S1-C01", "S2-C01"]) == "Sources: 2 of 3 · Evidence: S1-C01, S2-C01"   # only S1-S3 were readable
    assert position_indicator(b, ["S2-C01", "S1-C01"]) == "Sources: S1, S2 · Evidence: S2-C01, S1-C01"


def test_evidence_rows_form_the_chain_and_dedupe():
    rows = evidence_rows(_brief(), ["S2-C01", "S2-C01", "S1-C01"])
    assert [r["claim_id"] for r in rows] == ["S2-C01", "S1-C01"]
    assert rows[0]["url"] == "https://b.example/x" and rows[0]["quote"] == "quote for S2-C01"
