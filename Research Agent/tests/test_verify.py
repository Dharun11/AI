from research_agent.models import Claim, Source
from research_agent.pipeline.extract import chunk_text
from research_agent.pipeline.verify import is_grounded, verify_claims

TEXT = (
    "Online arbitration of consumer disputes in India takes on average under six months, "
    "according to platform data. By contrast, civil litigation commonly takes between three and five years."
)


def _claim(cid, quote, statement="x", src="S1"):
    return Claim(id=cid, source_id=src, statement=statement, quote=quote, facet="timeline")


def test_exact_and_typographic_quotes_are_grounded():
    assert is_grounded("takes on average under six months", TEXT, 85)
    assert is_grounded("civil   litigation commonly takes between three\nand five years", TEXT, 85)
    assert is_grounded("By contrast, civil litigation commonly takes between three and five years—", TEXT, 85)


def test_fabricated_quote_is_rejected():
    assert not is_grounded("AI-driven case allocation reduces bias by 15 percent", TEXT, 85)
    assert not is_grounded("six months", TEXT, 85)  # too short to count as evidence


def test_verify_drops_ungrounded_and_duplicates():
    src = Source(id="S1", url="https://a", text=TEXT)
    claims = [
        _claim("S1-C01", "takes on average under six months", "Arbitration takes under six months on average."),
        _claim("S1-C02", "reduces bias by 15 percent across all cases", "AI reduces bias by 15%."),
        _claim("S1-C03", "on average under six months, according", "Arbitration takes under six months on average"),
    ]
    kept, ungrounded, dupes = verify_claims(claims, [src], 85)
    assert [c.id for c in kept] == ["S1-C01"]
    assert ungrounded == 1 and dupes == 1


def test_chunking_covers_text_with_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(50))
    chunks = chunk_text(text, 2000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
    assert "Paragraph 0" in chunks[0] and "Paragraph 49" in chunks[-1]
