"""Grounding: a claim survives only if its quote is actually present in the scraped source text."""
import re
import unicodedata

from rapidfuzz import fuzz

from ..models import Claim, Source

_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP).lower()
    return re.sub(r"\s+", " ", text).strip()


def is_grounded(quote: str, source_text: str, threshold: int) -> bool:
    q, t = normalize(quote), normalize(source_text)
    if len(q) < 15:          # too short to be meaningful evidence
        return False
    if q in t:
        return True
    return fuzz.partial_ratio(q, t) >= threshold


def verify_claims(claims: list[Claim], sources: list[Source], threshold: int) -> tuple[list[Claim], int, int]:
    """Return (kept, n_ungrounded, n_duplicates)."""
    texts = {s.id: s.text for s in sources}
    grounded, ungrounded = [], 0
    for c in claims:
        if is_grounded(c.quote, texts.get(c.source_id, ""), threshold):
            grounded.append(c)
        else:
            ungrounded += 1

    # Chunk overlap can yield the same claim twice from one source; keep the first.
    kept: list[Claim] = []
    dupes = 0
    for c in grounded:
        if any(k.source_id == c.source_id and fuzz.token_set_ratio(k.statement, c.statement) >= 92 for k in kept):
            dupes += 1
            continue
        kept.append(c)
    return kept, ungrounded, dupes
