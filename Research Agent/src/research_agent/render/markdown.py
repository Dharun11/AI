"""Render a `Brief` to Markdown. All URLs and quotes come from code-held data, never from LLM text."""
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..config import get_settings
from ..models import Brief
from .view import evidence_rows, indicator, position_indicator, short_name, source_rows

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
_env.filters["cell"] = lambda s: str(s).replace("|", "\\|").replace("\n", " ")

OUTLIERS_SHOWN = 3  # per source, in extraction order (the extractor ranks by decision relevance)


def render_markdown(brief: Brief) -> str:
    sources = {s.id: s for s in brief.sources}
    cited: list[str] = []

    def cite(cid: str) -> str:
        """Inline source link plus the claim ID; the full quote is in the Evidence chain table."""
        c = brief.claims[cid]
        if cid not in cited:
            cited.append(cid)
        s = sources[c.source_id]
        return f"[{s.id} · {short_name(s)}]({s.url}) `{cid}`"

    outliers_by_source: dict[str, list[str]] = {}
    for cid in brief.outliers:
        outliers_by_source.setdefault(brief.claims[cid].source_id, []).append(cid)

    # The evidence table lists every claim cited in the body, so render the body first to collect them.
    template = _env.get_template("brief.md.j2")
    ctx = dict(
        brief=brief,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        sources_ok=sum(s.usable for s in brief.sources),
        claim=brief.claims.__getitem__,
        source=sources.__getitem__,
        cite=cite,
        indicator=lambda ids: indicator(brief, ids),
        position_indicator=lambda ids: position_indicator(brief, ids),
        outliers_by_source=sorted(outliers_by_source.items()),
        outliers_shown=OUTLIERS_SHOWN,
        source_rows=source_rows(brief),
        threshold=get_settings().grounding_threshold,
        evidence=lambda: evidence_rows(brief, cited),   # called at the end of the template, after every cite()
    )
    return template.render(**ctx)
