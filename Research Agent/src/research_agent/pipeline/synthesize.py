"""Cross-source reasoning. The LLM proposes groups by claim ID; code validates them and derives outliers."""
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..models import Claim, ClaimGroup, GroupOut, GroupsOut
from .trace import ask

log = logging.getLogger(__name__)

SYSTEM = """You compare factual claims extracted from several different sources on one research topic.
Each claim is tagged [S<source>-C<n>]; the S-number identifies which source made it.

Produce groups of two kinds:
1. "consensus": claims from TWO OR MORE DIFFERENT sources that assert substantially the same thing
   (same direction, compatible numbers). Claims from the same source alone never form consensus.
2. "contradiction": claims from different sources that cannot both be true, or that take explicitly
   opposing positions on the same question (e.g. "X is enforceable without court intervention" vs
   "courts frequently stay X"). Put each opposing side's claim IDs in its own list in `positions`.

Rules:
- Use ONLY claim IDs that appear in the list. Never invent IDs or sources.
- Differences in scope or emphasis are not contradictions; numbers that differ but overlap are not contradictions.
- A claim may appear in at most one consensus group, but may also appear in a contradiction.
- Do not group claims merely because they share a facet; they must make the same assertion.
- Leave unmatched claims out; they will be reported as single-source claims automatically."""


def format_claims(claims: list[Claim]) -> str:
    return "\n".join(f"[{c.id}] ({c.facet}) {c.statement}" for c in claims)


def validate_groups(
    groups: list[GroupOut], claims: list[Claim]
) -> tuple[list[ClaimGroup], list[ClaimGroup], list[str], int]:
    """Enforce citation rules in code. Return (consensus, contradictions, outlier_ids, n_rejected)."""
    by_id = {c.id: c for c in claims}
    consensus: list[ClaimGroup] = []
    contradictions: list[ClaimGroup] = []
    in_consensus: set[str] = set()
    rejected = 0

    for g in groups:
        if g.kind == "consensus":
            ids = [i for i in dict.fromkeys(g.claim_ids) if i in by_id and i not in in_consensus]
            if len({by_id[i].source_id for i in ids}) < 2:
                log.info("rejected consensus %r: ids %s -> valid unused %s (< 2 sources)", g.summary, g.claim_ids, ids)
                rejected += 1
                continue
            in_consensus.update(ids)
            consensus.append(ClaimGroup(kind="consensus", summary=g.summary, claim_ids=ids))
        else:
            positions = [[i for i in dict.fromkeys(p) if i in by_id] for p in g.positions]
            positions = [p for p in positions if p]
            side_sources = [{by_id[i].source_id for i in p} for p in positions]
            all_sources = set().union(*side_sources) if side_sources else set()
            # need >=2 sides, spanning >=2 sources, and sides must not be the same single source
            if len(positions) < 2 or len(all_sources) < 2:
                log.info("rejected contradiction %r: positions %s (need 2+ sides across 2+ sources)", g.summary, g.positions)
                rejected += 1
                continue
            ids = [i for p in positions for i in p]
            contradictions.append(ClaimGroup(kind="contradiction", summary=g.summary, claim_ids=ids, positions=positions))

    contradicted = {i for g in contradictions for i in g.claim_ids}
    outliers = [c.id for c in claims if c.id not in in_consensus and c.id not in contradicted]
    return consensus, contradictions, outliers, rejected


async def synthesize(llm: BaseChatModel, topic: str, claims: list[Claim]):
    if len({c.source_id for c in claims}) < 2:
        return validate_groups([], claims)
    out: GroupsOut = await ask(llm, GroupsOut, [
        SystemMessage(SYSTEM),
        HumanMessage(f"Research topic: {topic}\n\nClaims:\n{format_claims(claims)}"),
    ], label="synthesize", accept=lambda o: bool(o.groups))  # zero groups for many claims means the call misfired
    groups = out.groups if out else []
    consensus, contradictions, outliers, rejected = validate_groups(groups, claims)
    log.info("validate: LLM proposed %d groups -> kept %d consensus + %d contradictions, rejected %d; %d outliers",
             len(groups), len(consensus), len(contradictions), rejected, len(outliers))
    return consensus, contradictions, outliers, rejected
