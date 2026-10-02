"""The trace log must show every node and LLM call, and its topology table must match the real graph."""
import logging

import httpx
import pytest
import respx
from langchain_core.messages import HumanMessage

from research_agent.models import GroupOut, GroupsOut
from research_agent.pipeline import graph as graph_mod
from research_agent.pipeline.graph import NEXT, build_graph, run_research
from research_agent.pipeline.trace import EmptyAnswerError, ask, brief, describe

from .conftest import FakeLLM
from .test_api import PAGES, make_fake_llm


def test_next_table_matches_compiled_graph():
    edges = {(e.source, e.target) for e in build_graph(FakeLLM({})).get_graph().edges}
    declared = {("__start__", "fetch")} | {
        (src, "__end__" if dst == "END" else dst) for src, dsts in NEXT.items() for dst in dsts
    }
    assert edges == declared


def test_describe_and_brief_are_compact():
    from research_agent.models import ClaimOut, ClaimsOut
    out = ClaimsOut(claims=[ClaimOut(statement="Arbitration takes under six months.", quote="q" * 20, facet="timeline")])
    assert describe(out) == "claims[1] e.g. 'Arbitration takes under six months.'"
    assert describe(None).startswith("None")
    assert brief("x" * 2000) == "2,000 chars"


@respx.mock
async def test_run_logs_every_node_and_llm_call(caplog):
    for url, text in PAGES.items():
        respx.get(url).mock(return_value=httpx.Response(
            200, text=f"<html><body><article><p>{text}</p></article></body></html>", headers={"content-type": "text/html"}))
    caplog.set_level(logging.DEBUG, logger="research_agent")

    await run_research("Arbitration vs litigation in India", list(PAGES), llm=make_fake_llm())

    log_text = "\n".join(r.getMessage() for r in caplog.records)
    for node in ("fetch", "extract", "verify", "synthesize", "gaps", "tldr"):
        assert f"[node:{node}] START" in log_text and f"[node:{node}] DONE" in log_text
    assert "-> synthesize + gaps" in log_text            # parallel fan-out is visible
    for label in ("extract S1 1/1", "synthesize", "gaps: expected", "gaps: coverage", "tldr"):
        assert f">> LLM {label}" in log_text and f"<< LLM {label}" in log_text
    assert "answer [synthesize]" in log_text             # DEBUG shows the full intermediate answer
    assert "validate: LLM proposed 3 groups" in log_text
    assert "ignored 1 'covered' facets that cite no real claim" in log_text
    assert "run DONE" in log_text


async def test_ask_retries_a_missing_or_implausible_answer_then_succeeds(caplog):
    good = GroupsOut(groups=[GroupOut(kind="consensus", summary="s", claim_ids=["S1-C01", "S2-C01"])])
    answers = iter([None, GroupsOut(groups=[]), good])
    llm = FakeLLM({"GroupsOut": lambda messages: next(answers)})
    caplog.set_level(logging.WARNING, logger="research_agent")

    out = await ask(llm, GroupsOut, [HumanMessage("x")], "synthesize", accept=lambda o: bool(o.groups))

    assert out is good and len(llm.calls) == 3
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "no structured answer (attempt 1/3), retrying" in text
    assert "implausible answer (attempt 2/3), retrying" in text


async def test_ask_raises_instead_of_returning_a_silent_empty_result():
    llm = FakeLLM({"GroupsOut": None})
    with pytest.raises(EmptyAnswerError):
        await ask(llm, GroupsOut, [HumanMessage("x")], "synthesize")
    assert len(llm.calls) == 3


async def test_ask_returns_a_parsed_empty_answer_after_last_attempt():
    """If the model keeps saying 'no groups', that is its answer: return it (with warnings) rather than crash."""
    llm = FakeLLM({"GroupsOut": GroupsOut(groups=[])})
    out = await ask(llm, GroupsOut, [HumanMessage("x")], "synthesize", accept=lambda o: bool(o.groups))
    assert out.groups == [] and len(llm.calls) == 3
