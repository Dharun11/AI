"""LangGraph wiring: fetch -> extract -> verify -> (synthesize || gaps) -> tldr."""
import logging
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from ..fetch.fetcher import fetch_all
from ..llm import get_llm
from ..models import Brief, Claim, ClaimGroup, Gap, Source, Stats, TldrOut
from .extract import extract_claims
from .gaps import find_gaps
from .synthesize import synthesize
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


def build_graph(llm: BaseChatModel | None = None):
    llm = llm or get_llm()
    settings = get_settings()

    async def fetch(state: State) -> State:
        sources = await fetch_all(state["urls"])
        usable = [s for s in sources if s.status != "failed"]
        if len(usable) < 2:
            detail = "; ".join(f"{s.url}: {s.error}" for s in sources if s.status == "failed")
            raise InsufficientSourcesError(f"Only {len(usable)} source(s) could be read. {detail}")
        return {"sources": sources, "stats": Stats(sources_ok=len(usable), llm_provider=settings.llm_provider)}

    async def extract(state: State) -> State:
        claims = await extract_claims(llm, state["topic"], state["sources"])
        stats = state["stats"].model_copy(update={"claims_extracted": len(claims)})
        return {"claims": claims, "stats": stats}

    async def verify(state: State) -> State:
        kept, ungrounded, dupes = verify_claims(state["claims"], state["sources"], settings.grounding_threshold)
        log.info("verify: kept=%d ungrounded=%d dupes=%d", len(kept), ungrounded, dupes)
        stats = state["stats"].model_copy(
            update={"claims_rejected_ungrounded": ungrounded, "claims_deduplicated": dupes}
        )
        return {"claims": kept, "stats": stats}

    async def synth(state: State) -> State:
        consensus, contradictions, outliers, rejected = await synthesize(llm, state["topic"], state["claims"])
        stats = state["stats"].model_copy(update={"groups_rejected": rejected})
        return {"consensus": consensus, "contradictions": contradictions, "outliers": outliers, "stats": stats}

    async def gaps(state: State) -> State:
        return {"gaps": await find_gaps(llm, state["topic"], state["claims"])}

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
        out: TldrOut = await llm.with_structured_output(TldrOut).ainvoke([
            SystemMessage(TLDR_SYSTEM),
            HumanMessage(f"Topic: {state['topic']}\n\n{findings}"),
        ])
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


async def run_research(topic: str, urls: list[str], llm: BaseChatModel | None = None) -> Brief:
    state: State = await build_graph(llm).ainvoke({"topic": topic, "urls": urls})
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
