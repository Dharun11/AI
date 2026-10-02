"""Gap detection: expected facets of the topic that zero grounded claims address."""
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..models import Claim, CoverageOut, FacetsOut, Gap
from .trace import ask

log = logging.getLogger(__name__)

BASELINE_FACETS = [
    "Regulatory / compliance",
    "Cost / pricing",
    "Security / data privacy",
    "Scalability / volume handling",
    "Enforcement / legal risk",
    "Timelines / turnaround",
    "Stakeholder / end-user impact",
    "Accessibility / inclusion (language, geography, digital literacy)",
]

FACETS_SYSTEM = """You design research checklists for product managers, business analysts and legal specialists.
Given a research topic, list 8-12 dimensions ("facets") a decision-maker would expect a thorough brief to cover.
Start from this baseline, keeping the ones that apply to the topic and rephrasing them to be topic-specific:
{baseline}
Then add topic-specific facets (e.g. a specific statute, regulator, market segment, or metric).
Each facet name must be short (2-6 words). Give a one-sentence `why_it_matters` for each.
Do NOT look at any sources; this is the expected coverage, defined independently of what was found."""

COVERAGE_SYSTEM = """You audit research coverage. Given a list of expected facets and a list of claims extracted
from the sources, return each facet that at least one claim substantively addresses, together with the IDs of
those claims. A facet is covered only if a claim makes a concrete assertion about that specific facet; a claim that
is merely related to the general topic, or mentions a keyword in passing, does not count.
Copy facet names and claim IDs exactly as given. Omit facets that no claim addresses."""


def _key(s: str) -> str:
    return " ".join(s.lower().split())


async def find_gaps(llm: BaseChatModel, topic: str, claims: list[Claim]) -> list[Gap]:
    facets_out: FacetsOut = await ask(llm, FacetsOut, [
        SystemMessage(FACETS_SYSTEM.format(baseline="\n".join(f"- {b}" for b in BASELINE_FACETS))),
        HumanMessage(f"Research topic: {topic}"),
    ], label="gaps: expected")
    expected = facets_out.facets if facets_out else []
    if not expected:
        return []
    if not claims:
        return expected

    facet_list = "\n".join(f"- {f.facet}" for f in expected)
    claim_list = "\n".join(f"[{c.id}] ({c.facet}) {c.statement}" for c in claims)
    cov: CoverageOut = await ask(llm, CoverageOut, [
        SystemMessage(COVERAGE_SYSTEM),
        HumanMessage(f"Research topic: {topic}\n\nExpected facets:\n{facet_list}\n\nClaims:\n{claim_list}"),
    ], label="gaps: coverage")
    # A facet counts as covered only when the auditor cites at least one real claim for it.
    known = {c.id for c in claims}
    claimed = cov.covered if cov else []
    covered = {_key(fc.facet) for fc in claimed if any(i in known for i in fc.claim_ids)}
    unsupported = [fc.facet for fc in claimed if _key(fc.facet) not in covered]
    if unsupported:
        log.info("coverage: ignored %d 'covered' facets that cite no real claim: %s", len(unsupported), unsupported)
    gaps = [f for f in expected if _key(f.facet) not in covered]
    log.info("coverage: %d expected facets, %d covered by evidence -> %d gaps", len(expected), len(covered), len(gaps))
    return gaps
