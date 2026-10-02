"""LangGraph wiring: fetch -> extract -> verify -> (synthesize || gaps) -> tldr."""
import logging
import time
import uuid
from collections.abc import Callable
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from ..fetch.fetcher import fetch_all
from ..llm import get_llm
from ..models import Brief, Claim, ClaimGroup, Gap, Source, Stats, TldrOut
from .extract import apply_coverage, extract_claims
from .gaps import find_gaps
from .synthesize import synthesize
from .trace import ask, progress_var, run_id_var, traced_node
from .verify import verify_claims

log = logging.getLogger(__name__)

TLDR_SYSTEM = """Write a 3-bullet TL;DR for a decision-maker who has two minutes.
Use ONLY the findings provided; do not add facts. Lead with the strongest consensus, then the most
material contradiction or risk, then the most important gap. If a category says "none", do not write
about it or invent one; use that bullet for the next most decision-relevant finding instead.
No citations or source names needed."""


class InsufficientSourcesError(RuntimeError):
    """Fewer than two usable sources: cross-source reasoning is impossible."""


class State(TypedDict, total=False):
    topic: str
    urls: list[str]
    sources: list[Source]
    claims: list[Claim]
    consensus: list[ClaimGroup]
    contradictions: list[ClaimGroup]
    outliers: list[str]
    gaps: list[Gap]
    tldr: list[str]
    stats: Stats


# Where control goes after each node. Used for the trace log; a test checks it against the compiled graph.
NEXT = {
    "fetch": ["extract"],
    "extract": ["verify"],
    "verify": ["synthesize", "gaps"],     # parallel
    "synthesize": ["tldr"],               # tldr waits for both
    "gaps": ["tldr"],
    "tldr": ["END"],
}


def build_graph(llm: BaseChatModel | None = None):
    llm = llm or get_llm()
    settings = get_settings()

    @traced_node("fetch", NEXT["fetch"])
    async def fetch(state: State) -> State:
        sources = await fetch_all(state["urls"])
        for s in sources:
            log.info("fetched %s %-7s %-12s %7s chars  %s%s", s.id, s.status, s.method or "-", f"{len(s.text):,}",
                     s.url, f"  ({s.error})" if s.error else "")
        usable = [s for s in sources if s.usable]
        if len(usable) < 2:
            detail = "; ".join(f"{s.url}: {s.status} ({s.error})" for s in sources if not s.usable)
            raise InsufficientSourcesError(f"Only {len(usable)} source(s) could be read. {detail}")
        return {"sources": sources, "stats": Stats(sources_ok=len(usable), llm_provider=settings.llm_provider)}

    @traced_node("extract", NEXT["extract"])
    async def extract(state: State) -> State:
        claims, coverage = await extract_claims(llm, state["topic"], state["sources"])
        sources = apply_coverage(state["sources"], coverage)        # flags sources whose chunks yielded nothing
        stats = state["stats"].model_copy(update={"claims_extracted": len(claims)})
        return {"claims": claims, "sources": sources, "stats": stats}

    @traced_node("verify", NEXT["verify"])
    async def verify(state: State) -> State:
        kept, ungrounded, dupes = verify_claims(state["claims"], state["sources"], settings.grounding_threshold)
        log.info("verify: %d claims in -> %d kept, %d ungrounded (quote not in source), %d duplicates",
                 len(state["claims"]), len(kept), ungrounded, dupes)
        stats = state["stats"].model_copy(
            update={"claims_rejected_ungrounded": ungrounded, "claims_deduplicated": dupes}
        )
        return {"claims": kept, "stats": stats}

    @traced_node("synthesize", NEXT["synthesize"])
    async def synth(state: State) -> State:
        consensus, contradictions, outliers, rejected = await synthesize(llm, state["topic"], state["claims"])
        stats = state["stats"].model_copy(update={"groups_rejected": rejected})
        return {"consensus": consensus, "contradictions": contradictions, "outliers": outliers, "stats": stats}

    @traced_node("gaps", NEXT["gaps"])
    async def gaps(state: State) -> State:
        return {"gaps": await find_gaps(llm, state["topic"], state["claims"])}

    @traced_node("tldr", NEXT["tldr"])
    async def tldr(state: State) -> State:
        by_id = {c.id: c for c in state["claims"]}
        def section(title: str, lines: list[str]) -> str:
            return f"{title}:\n" + ("\n".join(f"- {line}" for line in lines) if lines else "- none")

        findings = "\n\n".join([
            section("Consensus", [f"{g.summary} ({len({by_id[i].source_id for i in g.claim_ids})} sources)"
                                  for g in state["consensus"]]),
            section("Contradictions", [g.summary for g in state["contradictions"]]),
            section("Gaps (no source covers)", [g.facet for g in state["gaps"]]),
            section("Single-source claims (sample)", [by_id[i].statement for i in state["outliers"][:15]]),
        ])
        out: TldrOut = await ask(llm, TldrOut, [
            SystemMessage(TLDR_SYSTEM),
            HumanMessage(f"Topic: {state['topic']}\n\n{findings}"),
        ], label="tldr")
        return {"tldr": (out.bullets if out else [])[:3]}

    g = StateGraph(State)
    g.add_node("fetch", fetch)
    g.add_node("extract", extract)
    g.add_node("verify", verify)
    g.add_node("synthesize", synth)
    g.add_node("gaps", gaps)
    g.add_node("tldr", tldr)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "extract")
    g.add_edge("extract", "verify")
    g.add_edge("verify", "synthesize")   # synthesize and gaps run in parallel
    g.add_edge("verify", "gaps")
    g.add_edge(["synthesize", "gaps"], "tldr")
    g.add_edge("tldr", END)
    return g.compile()


async def run_research(topic: str, urls: list[str], llm: BaseChatModel | None = None,
                       progress: Callable[[str], None] | None = None) -> Brief:
    """`progress`, if given, receives a short text line as each node and LLM call finishes."""
    run_id_var.set(uuid.uuid4().hex[:6])
    progress_var.set(progress)
    log.info("run START topic=%r, %d urls, provider=%s", topic, len(urls), get_settings().llm_provider)
    t0 = time.perf_counter()
    state: State = await build_graph(llm).ainvoke({"topic": topic, "urls": urls})
    log.info("run DONE %.1fs: %d consensus, %d contradictions, %d outliers, %d gaps",
             time.perf_counter() - t0, len(state["consensus"]), len(state["contradictions"]),
             len(state["outliers"]), len(state["gaps"]))
    return Brief(
        topic=topic,
        sources=state["sources"],
        claims={c.id: c for c in state["claims"]},
        tldr=state.get("tldr", []),
        consensus=state.get("consensus", []),
        contradictions=state.get("contradictions", []),
        outliers=state.get("outliers", []),
        gaps=state.get("gaps", []),
        stats=state["stats"],
    )
