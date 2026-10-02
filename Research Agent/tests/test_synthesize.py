from research_agent.models import Claim, GroupOut
from research_agent.pipeline.synthesize import validate_groups


def _c(cid):
    return Claim(id=cid, source_id=cid.split("-")[0], statement=cid, quote="q" * 20, facet="f")


CLAIMS = [_c(i) for i in ["S1-C01", "S1-C02", "S2-C01", "S2-C02", "S3-C01", "S4-C01"]]


def test_valid_consensus_and_contradiction_kept():
    groups = [
        GroupOut(kind="consensus", summary="fast", claim_ids=["S1-C01", "S2-C01"]),
        GroupOut(kind="contradiction", summary="enforceable?", claim_ids=["S1-C02", "S4-C01"],
                 positions=[["S1-C02"], ["S4-C01"]]),
    ]
    consensus, contradictions, outliers, rejected = validate_groups(groups, CLAIMS)
    assert [g.claim_ids for g in consensus] == [["S1-C01", "S2-C01"]]
    assert contradictions[0].positions == [["S1-C02"], ["S4-C01"]]
    assert outliers == ["S2-C02", "S3-C01"]
    assert rejected == 0


def test_single_source_consensus_is_demoted_to_outliers():
    groups = [GroupOut(kind="consensus", summary="same source twice", claim_ids=["S1-C01", "S1-C02"])]
    consensus, _, outliers, rejected = validate_groups(groups, CLAIMS)
    assert consensus == [] and rejected == 1
    assert "S1-C01" in outliers and "S1-C02" in outliers


def test_hallucinated_ids_are_dropped():
    groups = [
        # S9 does not exist -> only S3 remains -> not consensus
        GroupOut(kind="consensus", summary="x", claim_ids=["S3-C01", "S9-C01"]),
        # one real id survives alongside a fake one; still spans 2 sources
        GroupOut(kind="consensus", summary="y", claim_ids=["S1-C01", "S2-C01", "S7-C99"]),
    ]
    consensus, _, outliers, rejected = validate_groups(groups, CLAIMS)
    assert rejected == 1
    assert consensus[0].claim_ids == ["S1-C01", "S2-C01"]
    assert "S3-C01" in outliers


def test_contradiction_needs_two_sides_from_different_sources():
    groups = [
        GroupOut(kind="contradiction", summary="self", claim_ids=[], positions=[["S1-C01"], ["S1-C02"]]),
        GroupOut(kind="contradiction", summary="one side", claim_ids=[], positions=[["S1-C01", "S2-C01"]]),
        GroupOut(kind="contradiction", summary="no positions", claim_ids=["S1-C01", "S2-C01"]),
    ]
    _, contradictions, _, rejected = validate_groups(groups, CLAIMS)
    assert contradictions == [] and rejected == 3


def test_claim_not_reused_across_consensus_groups():
    groups = [
        GroupOut(kind="consensus", summary="a", claim_ids=["S1-C01", "S2-C01"]),
        GroupOut(kind="consensus", summary="b", claim_ids=["S1-C01", "S3-C01"]),  # S1-C01 already used -> only S3
    ]
    consensus, _, _, rejected = validate_groups(groups, CLAIMS)
    assert len(consensus) == 1 and rejected == 1
