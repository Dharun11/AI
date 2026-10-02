"""Domain schemas. LLM-facing schemas (suffix `Out`) only ever reference IDs, never URLs."""
from typing import Literal

from pydantic import BaseModel, Field

SourceStatus = Literal["ok", "partial", "failed"]


class Source(BaseModel):
    id: str                      # S1..S5
    url: str
    title: str = ""
    status: SourceStatus = "ok"
    text: str = ""
    error: str | None = None
    method: str = ""             # how the text was obtained: trafilatura / bs4 / playwright / pypdf


class Claim(BaseModel):
    id: str                      # S2-C07
    source_id: str
    statement: str
    quote: str
    facet: str


class ClaimGroup(BaseModel):
    kind: Literal["consensus", "contradiction"]
    summary: str
    claim_ids: list[str]
    # contradiction only: each side is a list of claim ids taking that position
    positions: list[list[str]] = Field(default_factory=list)


class Gap(BaseModel):
    facet: str
    why_it_matters: str


class Stats(BaseModel):
    sources_ok: int = 0
    claims_extracted: int = 0
    claims_rejected_ungrounded: int = 0
    claims_deduplicated: int = 0
    groups_rejected: int = 0
    llm_provider: str = ""


class Brief(BaseModel):
    topic: str
    sources: list[Source]
    claims: dict[str, Claim]
    tldr: list[str] = Field(default_factory=list)
    consensus: list[ClaimGroup] = Field(default_factory=list)
    contradictions: list[ClaimGroup] = Field(default_factory=list)
    outliers: list[str] = Field(default_factory=list)   # claim ids
    gaps: list[Gap] = Field(default_factory=list)
    stats: Stats = Field(default_factory=Stats)


# ---------- LLM structured-output schemas ----------

class ClaimOut(BaseModel):
    statement: str = Field(description="One atomic, checkable factual claim, paraphrased in a full sentence.")
    quote: str = Field(description="Exact verbatim span copied from the source text that supports the claim (max ~300 chars).")
    facet: str = Field(description="1-3 word lowercase dimension of the topic, e.g. 'cost', 'enforceability', 'timeline'.")


class ClaimsOut(BaseModel):
    claims: list[ClaimOut]


class GroupOut(BaseModel):
    kind: Literal["consensus", "contradiction"]
    summary: str = Field(description="One-sentence neutral statement of the shared claim, or of the point of conflict.")
    claim_ids: list[str] = Field(description="All claim IDs involved, e.g. ['S1-C03', 'S2-C01'].")
    positions: list[list[str]] = Field(
        default_factory=list,
        description="Contradiction only: claim IDs grouped per opposing position, e.g. [['S1-C02'], ['S4-C05']].",
    )


class GroupsOut(BaseModel):
    groups: list[GroupOut]


class FacetsOut(BaseModel):
    facets: list[Gap] = Field(description="Expected dimensions of the topic, each with why a decision-maker needs it.")


class FacetCoverage(BaseModel):
    facet: str = Field(description="Expected facet name, copied exactly.")
    claim_ids: list[str] = Field(description="IDs of the claims that substantively address this facet.")


class CoverageOut(BaseModel):
    covered: list[FacetCoverage] = Field(description="Only facets that at least one claim substantively addresses.")


class TldrOut(BaseModel):
    bullets: list[str] = Field(description="Exactly 3 decision-oriented bullets, each under 30 words.")
