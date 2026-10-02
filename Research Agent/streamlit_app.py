"""Streamlit UI: enter a topic and 3-5 URLs, watch the pipeline run, read and download the brief.

Run with:  uv run streamlit run streamlit_app.py
"""
import asyncio
import os
import re
import time
from urllib.parse import urlparse

import streamlit as st

from research_agent.config import get_settings
from research_agent.fetch.js import INSTALL_HINT, browser_installed
from research_agent.llm import LLMConfigError
from research_agent.models import Brief
from research_agent.pipeline.graph import InsufficientSourcesError, run_research
from research_agent.pipeline.trace import configure_logging
from research_agent.render.markdown import OUTLIERS_SHOWN, render_markdown
from research_agent.render.view import COLOR, evidence_rows, indicator, position_indicator, short_name, source_rows

st.set_page_config(page_title="Research Synthesis Agent", page_icon=None, layout="wide")
configure_logging()

MIN_URLS, MAX_URLS = 3, 5
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
           "gemini": "GOOGLE_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
EXAMPLE_TOPIC = "Impact of AI on jobs and the labor market"
EXAMPLE_URLS = "\n".join([
    "https://insights.som.yale.edu/insights/the-real-job-destruction-from-ai-is-hitting-before-careers-can-start",
    "https://www.library.hbs.edu/working-knowledge/enhance-or-eliminate-how-ai-will-likely-change-these-jobs",
    "https://gloat.com/blog/ai-labor-market/",
    "https://www.brookings.edu/articles/measuring-us-workers-capacity-to-adapt-to-ai-driven-job-displacement/",
    "https://www.jpmorgan.com/insights/global-research/artificial-intelligence/ai-impact-job-growth",
])


def parse_urls(text: str) -> tuple[list[str], list[str]]:
    """Split on whitespace or commas, drop duplicates, and report anything that isn't a usable http(s) URL."""
    urls, errors, seen = [], [], set()
    for token in re.split(r"[\s,]+", text.strip()):
        if not token:
            continue
        parsed = urlparse(token)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            errors.append(f"Not a valid http(s) URL: {token}")
        elif token.rstrip("/") in seen:
            errors.append(f"Duplicate URL: {token}")
        else:
            seen.add(token.rstrip("/"))
            urls.append(token)
    return urls, errors


def esc(text: str) -> str:
    """Streamlit treats $...$ as LaTeX; source text is full of dollar amounts."""
    return text.replace("$", r"\$")


def claim_md(brief: Brief, cid: str, quote: bool = False) -> str:
    """One claim as a bullet: statement, linked source, claim ID (and optionally the exact quote)."""
    c = brief.claims[cid]
    s = next(s for s in brief.sources if s.id == c.source_id)
    line = f"- {esc(c.statement)} [{s.id} · {short_name(s)}]({s.url}) `{cid}`"
    return line + (f"\n  - *“{esc(' '.join(c.quote.split()))}”*" if quote else "")


def claims_md(brief: Brief, ids: list[str], quote: bool = False) -> str:
    return "\n".join(claim_md(brief, cid, quote) for cid in ids)


def evidence_chain(brief: Brief, ids: list[str]) -> None:
    """Finding -> claim ID -> source -> exact quote -> URL, as a table that wraps long quotes."""
    rows = evidence_rows(brief, ids)
    cell = lambda t: esc(t).replace("|", "\\|")  # noqa: E731
    table = ["| Claim | Source | Exact quote | URL |", "|---|---|---|---|"] + [
        f"| `{r['claim_id']}` | {r['source_id']} {cell(r['name'])} | “{cell(r['quote'])}” | {r['url']} |" for r in rows]
    with st.expander(f"Evidence chain ({len(rows)} claim{'s' if len(rows) != 1 else ''})"):
        st.markdown("\n".join(table))


def render_source_strip(brief: Brief) -> None:
    """One line per source showing what happened to it: the graceful-failure view."""
    parts = [f":{COLOR[r['status']]}[{r['icon']} **{r['name']}**] {esc(r['label'])}" for r in source_rows(brief)]
    st.markdown("  \n".join(parts))


def render_brief(brief: Brief) -> None:
    st.subheader("TL;DR")
    for bullet in brief.tldr or ["No findings could be summarised."]:
        st.markdown(f"- {esc(bullet)}")

    st.subheader(f"Consensus ({len(brief.consensus)})", help="Two or more different sources make the same assertion.")
    if not brief.consensus:
        st.caption("No claim was corroborated by two or more independent sources.")
    for i, g in enumerate(brief.consensus, 1):
        with st.container(border=True):
            st.markdown(f"**Consensus {i}: {esc(g.summary)}**")
            st.caption(indicator(brief, g.claim_ids))
            st.markdown(claims_md(brief, g.claim_ids))
            evidence_chain(brief, g.claim_ids)

    st.subheader(f"Contradictions ({len(brief.contradictions)})", help="Claims from different sources that cannot both be true.")
    if not brief.contradictions:
        st.caption("No direct contradictions between sources were found.")
    for i, g in enumerate(brief.contradictions, 1):
        with st.container(border=True):
            st.markdown(f"**Contradiction {i}: {esc(g.summary)}**")
            for n, side in enumerate(g.positions):
                if n:
                    st.markdown("<div style='text-align:center;font-weight:600;opacity:.7'>vs.</div>", unsafe_allow_html=True)
                label = g.position_labels[n] if n < len(g.position_labels) else brief.claims[side[0]].statement
                st.markdown(f"**Position {'ABCDEFG'[n]}: {esc(label)}**")
                st.caption(position_indicator(brief, side))
                st.markdown(claims_md(brief, side))
            evidence_chain(brief, g.claim_ids)

    st.subheader(f"Single-source claims ({len(brief.outliers)})",
                 help="Verified claims made by only one source and not contested. Not yet corroborated.")
    by_source: dict[str, list[str]] = {}
    for cid in brief.outliers:
        by_source.setdefault(brief.claims[cid].source_id, []).append(cid)
    titles = {s.id: s.title for s in brief.sources}
    for sid, ids in sorted(by_source.items()):
        st.markdown(f"**{sid}: {esc(titles[sid])}**")
        st.markdown(claims_md(brief, ids[:OUTLIERS_SHOWN], quote=True))
        if len(ids) > OUTLIERS_SHOWN:
            with st.expander(f"{len(ids) - OUTLIERS_SHOWN} more from {sid}"):
                st.markdown(claims_md(brief, ids[OUTLIERS_SHOWN:], quote=True))
    if not by_source:
        st.caption("Every claim was corroborated or contested by another source.")

    st.subheader(f"Gaps ({len(brief.gaps)})", help="Topics a decision-maker would expect that no source addresses.")
    if not brief.gaps:
        st.caption("All expected facets of the topic were covered by at least one source.")
    for gap in brief.gaps:
        st.markdown(f"- **{esc(gap.facet)}**: {esc(gap.why_it_matters)}")

    with st.expander("Methodology"):
        s = brief.stats
        st.markdown(
            f"- Claims were extracted per source by `{s.llm_provider}`, each with a verbatim quote that was checked against the page text.\n"
            f"- {s.claims_extracted} claims extracted, {s.claims_rejected_ungrounded} rejected as ungrounded, "
            f"{s.claims_deduplicated} duplicates removed.\n"
            f"- {s.groups_rejected} proposed groups were discarded because they did not span 2 or more distinct sources.\n"
            "- Gaps are expected facets generated from the topic alone; a facet only counts as covered if real claims are cited for it."
        )


def render_source_status(brief: Brief) -> None:
    st.dataframe(
        [{"": r["icon"], "ID": r["id"], "Source": r["name"], "Status": r["label"], "Parsed with": r["method"],
          "Title": r["title"], "URL": r["url"]} for r in source_rows(brief)],
        hide_index=True, width="stretch",
        column_config={"URL": st.column_config.LinkColumn("URL")},
    )
    st.caption("A page that refuses our fetcher is shown as inaccessible; that does not prove a paywall, and nothing is bypassed. "
               "Inaccessible, timed-out and unparseable sources are skipped. "
               "“browser-rendered” means the plain download had too little text, so the page was opened in headless Chromium.")


def file_name(topic: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60] or "brief") + ".md"


# ---------------------------------------------------------------- sidebar
settings = get_settings()
with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Provider:** `{settings.llm_provider or 'not set'}`")
    st.markdown(f"**Model:** `{settings.llm_model.strip() or 'not set'}`")
    st.markdown(f"**Reasoning:** `{settings.llm_reasoning or 'model default'}`")
    if not settings.llm_provider or not settings.llm_model.strip():
        st.error("Set `LLM_PROVIDER` and `LLM_MODEL` in `.env`. No model is built in.")
    elif not os.getenv(KEY_ENV.get(settings.llm_provider, ""), ""):
        st.warning(f"`{KEY_ENV.get(settings.llm_provider, 'API key')}` is not set. Add it to `.env`.")
    st.caption("Change these in `.env` (LLM_PROVIDER, LLM_MODEL, LLM_REASONING) and restart the app.")
    st.divider()
    if not settings.use_playwright:
        st.markdown("**Browser rendering:** off (`USE_PLAYWRIGHT=false`)")
    elif browser_installed():
        st.markdown("**Browser rendering:** on, Chromium ready")
        st.caption("Pages with too little text are re-opened in a browser. Source status shows which ones.")
    else:
        st.warning(f"Browser rendering is on but Chromium is not installed, so JavaScript pages will be partial. {INSTALL_HINT.capitalize()}")
    st.divider()
    st.markdown("**How it works**")
    st.caption(
        "1. Fetch and clean each page or PDF\n\n2. Extract atomic claims with a supporting quote\n\n"
        "3. Verify each quote exists in the source\n\n4. Compare across sources, and find gaps\n\n5. Write the brief"
    )

# ---------------------------------------------------------------- input
st.title("Research Synthesis Agent")
st.caption(f"Enter a topic and {MIN_URLS} to {MAX_URLS} source URLs. The agent finds where sources agree, "
           "disagree, stand alone, or leave gaps. Every finding links back to its source with a verified quote.")

if st.button("Load example"):
    st.session_state["topic"] = EXAMPLE_TOPIC
    st.session_state["urls_text"] = EXAMPLE_URLS

with st.form("research_form"):
    topic = st.text_input("Topic", key="topic", placeholder="e.g. Effectiveness of arbitration vs litigation in consumer disputes")
    urls_text = st.text_area("Source URLs (one per line)", key="urls_text", height=150,
                             placeholder="https://example.com/article\nhttps://example.org/report.pdf\nhttps://example.net/post")
    submitted = st.form_submit_button("Run research", type="primary")

if submitted:
    urls, errors = parse_urls(urls_text)
    if len(topic.strip()) < 3:
        errors.append("Enter a topic of at least 3 characters.")
    if not MIN_URLS <= len(urls) <= MAX_URLS:
        errors.append(f"Enter {MIN_URLS} to {MAX_URLS} unique URLs (you have {len(urls)}).")
    if errors:
        for message in errors:
            st.error(message)
    else:
        st.session_state.pop("result", None)
        started = time.perf_counter()
        with st.status("Running pipeline...", expanded=True) as status:
            try:
                brief = asyncio.run(run_research(topic.strip(), urls, progress=lambda line: status.text(line)))
            except InsufficientSourcesError as e:
                status.update(label="Not enough readable sources", state="error")
                st.error(f"{e} The agent needs at least 2 readable sources to compare.")
            except LLMConfigError as e:
                status.update(label="LLM settings are incomplete", state="error")
                st.error(str(e))
            except Exception as e:  # surface API-key, network and provider errors instead of a blank page
                status.update(label="The run failed", state="error")
                st.error(f"{type(e).__name__}: {e}")
            else:
                elapsed = time.perf_counter() - started
                status.update(label=f"Done in {elapsed:.0f}s", state="complete", expanded=False)
                st.session_state["result"] = {"brief": brief, "markdown": render_markdown(brief), "elapsed": elapsed}

# ---------------------------------------------------------------- results
result = st.session_state.get("result")
if result:
    brief: Brief = result["brief"]
    st.divider()
    st.header(brief.topic)

    cols = st.columns(6)
    cols[0].metric("Sources processed", f"{brief.stats.sources_ok}/{len(brief.sources)}")
    cols[1].metric("Verified claims", len(brief.claims))
    cols[2].metric("Consensus", len(brief.consensus))
    cols[3].metric("Contradictions", len(brief.contradictions))
    cols[4].metric("Single-source", len(brief.outliers))
    cols[5].metric("Gaps", len(brief.gaps))

    render_source_strip(brief)
    st.download_button("Download brief (.md)", result["markdown"], file_name=file_name(brief.topic), mime="text/markdown")

    tab_brief, tab_sources, tab_md = st.tabs(["Brief", "Source status", "Markdown"])
    with tab_brief:
        render_brief(brief)
    with tab_sources:
        render_source_status(brief)
    with tab_md:
        st.code(result["markdown"], language="markdown", wrap_lines=True)
