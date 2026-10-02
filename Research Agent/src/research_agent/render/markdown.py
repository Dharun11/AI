"""Render a `Brief` to Markdown. All URLs and quotes come from code-held data, never from LLM text."""
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..config import get_settings
from ..models import Brief

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
_env.filters["cell"] = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
_env.filters["oneline"] = lambda s: " ".join(str(s).split()).replace('"', "'")

STATUS_ICON = {"ok": "🟢", "partial": "🟡", "failed": "🔴"}
OUTLIERS_SHOWN = 3  # per source, in extraction order (the extractor ranks by decision relevance)


def render_markdown(brief: Brief) -> str:
    sources = {s.id: s for s in brief.sources}
    cited: list[str] = []

    def cite(cid: str) -> str:
        c = brief.claims[cid]
        if cid not in cited:
            cited.append(cid)
        return f"[[{c.source_id}]]({sources[c.source_id].url})[^{cid}]"

    outliers_by_source: dict[str, list[str]] = {}
    for cid in brief.outliers:
        outliers_by_source.setdefault(brief.claims[cid].source_id, []).append(cid)

    # Evidence footnotes are rendered after the body, so render body first to populate `cited`.
    template = _env.get_template("brief.md.j2")
    ctx = dict(
        brief=brief,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        sources_ok=sum(s.status != "failed" for s in brief.sources),
        status_icon=STATUS_ICON,
        claim=brief.claims.__getitem__,
        source=sources.__getitem__,
        cite=cite,
        cited=cited,
        outliers_by_source=sorted(outliers_by_source.items()),
        outliers_shown=OUTLIERS_SHOWN,
        threshold=get_settings().grounding_threshold,
    )
    return template.render(**ctx)
