"""Presentation helpers shared by the Markdown template and the Streamlit app.

Pure functions of a `Brief`: every URL, source name and quote comes from data held in code, never from LLM text.
"""
from urllib.parse import urlparse

from ..models import Brief, Source

ICON = {"ok": "✓", "partial": "⚠", "blocked": "⚠", "timeout": "⚠", "parse_failed": "✗", "failed": "✗"}
COLOR = {"ok": "green", "partial": "orange", "blocked": "orange", "timeout": "orange", "parse_failed": "red", "failed": "red"}
# "blocked" says our fetcher was refused. That is not proof of a paywall, so the wording stays neutral.
STATUS_WORD = {"partial": "partial", "blocked": "inaccessible", "timeout": "timed out",
               "parse_failed": "could not be parsed", "failed": "failed"}


def short_name(source: Source) -> str:
    host = urlparse(source.url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def status_label(source: Source) -> str:
    """Plain words for what happened to a source, including whether a browser was needed."""
    if source.status == "ok":
        notes = (["browser-rendered"] if source.method == "playwright" else []) + ([source.error] if source.error else [])
        return "processed" + (f" ({'; '.join(notes)})" if notes else "")
    return STATUS_WORD[source.status] + (f" ({source.error})" if source.error else "")


def source_rows(brief: Brief) -> list[dict]:
    return [{"id": s.id, "name": short_name(s), "title": s.title, "url": s.url, "status": s.status,
             "icon": ICON[s.status], "label": status_label(s), "method": s.method or "-"} for s in brief.sources]


def source_ids(brief: Brief, claim_ids: list[str]) -> list[str]:
    """Distinct source IDs behind some claims, in S1, S2... order."""
    return sorted({brief.claims[c].source_id for c in claim_ids}, key=lambda sid: int(sid[1:]))


def indicator(brief: Brief, claim_ids: list[str]) -> str:
    """e.g. 'Sources: 2 of 5 · Evidence: S1-C02, S5-C01'. Honest counts, not an invented confidence score."""
    read = sum(s.usable for s in brief.sources)
    return f"Sources: {len(source_ids(brief, claim_ids))} of {read} · Evidence: {', '.join(claim_ids)}"


def position_indicator(brief: Brief, claim_ids: list[str]) -> str:
    return f"Sources: {', '.join(source_ids(brief, claim_ids))} · Evidence: {', '.join(claim_ids)}"


def evidence_rows(brief: Brief, claim_ids: list[str]) -> list[dict]:
    """The chain for each claim: claim ID -> source -> exact quote -> URL."""
    sources = {s.id: s for s in brief.sources}
    rows = []
    for cid in dict.fromkeys(claim_ids):
        c = brief.claims[cid]
        s = sources[c.source_id]
        rows.append({"claim_id": cid, "source_id": s.id, "name": short_name(s), "title": s.title, "url": s.url,
                     "statement": c.statement, "quote": " ".join(c.quote.split())})
    return rows
