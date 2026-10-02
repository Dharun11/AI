"""Per-source claim extraction. One structured-output call per text chunk, all chunks concurrent."""
import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..models import Claim, ClaimsOut, Source
from .trace import ask

SYSTEM = """You are a meticulous research analyst extracting factual claims from ONE source document.

Rules:
- Extract only claims relevant to the research topic.
- Each claim must be atomic (one assertion) and checkable: a fact, statistic, finding, legal position, or explicit argument the source makes.
- Skip navigation text, ads, author bios, calls to action, and vague marketing fluff with no concrete assertion.
- `quote` MUST be copied character-for-character from the source text (a contiguous span, max ~300 chars). Never paraphrase the quote, never stitch distant fragments together.
- `statement` is a self-contained paraphrase understandable without the source (resolve pronouns, keep numbers and qualifiers like "up to", "on average").
- `facet` is a short lowercase dimension label (e.g. "cost", "timeline", "enforceability", "data privacy").
- Return at most {max_claims} claims, prioritising the most decision-relevant. Return an empty list if nothing is relevant."""

MAX_CLAIMS_PER_CHUNK = 20


def chunk_text(text: str, size: int, overlap: int = 500) -> list[str]:
    """Split on paragraph boundaries where possible, with a small overlap so claims aren't cut in half."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind("\n\n", start + size // 2, end)
            end = cut if cut != -1 else end
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


async def _extract_chunk(llm: BaseChatModel, topic: str, src: Source, chunk: str, idx: int, total: int) -> list:
    msg = (
        f"Research topic: {topic}\n"
        f"Source title: {src.title}\n"
        f"Part {idx}/{total} of the source text:\n<source>\n{chunk}\n</source>"
    )
    out: ClaimsOut = await ask(
        llm, ClaimsOut,
        [SystemMessage(SYSTEM.format(max_claims=MAX_CLAIMS_PER_CHUNK)), HumanMessage(msg)],
        label=f"extract {src.id} {idx}/{total}",
    )
    return out.claims if out else []


async def extract_claims(llm: BaseChatModel, topic: str, sources: list[Source], concurrency: int = 4) -> list[Claim]:
    s = get_settings()
    sem = asyncio.Semaphore(concurrency)

    async def run(src: Source, chunk: str, i: int, n: int):
        async with sem:
            return src, await _extract_chunk(llm, topic, src, chunk, i, n)

    tasks = []
    for src in sources:
        if src.status == "failed":
            continue
        chunks = chunk_text(src.text, s.chunk_chars)
        tasks += [run(src, c, i, len(chunks)) for i, c in enumerate(chunks, 1)]

    results = await asyncio.gather(*tasks)

    claims: list[Claim] = []
    counters: dict[str, int] = {}
    for src, outs in results:
        for o in outs:
            counters[src.id] = counters.get(src.id, 0) + 1
            claims.append(Claim(
                id=f"{src.id}-C{counters[src.id]:02d}",
                source_id=src.id,
                statement=o.statement.strip(),
                quote=o.quote.strip(),
                facet=o.facet.strip().lower(),
            ))
    return claims
