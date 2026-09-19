"""LangGraph ReAct orchestrator: reasoner <-> tools loop, cloud LLM, per-step audit log."""
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.agent_tools import ALL_TOOLS, get_engine  # noqa: E402

SYSTEM_PROMPT = SystemMessage((ROOT / "src" / "prompts" / "system_prompt.txt").read_text(encoding="utf-8"))


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_llm():
    """Cloud LLM selected by LLM_PROVIDER (openai | anthropic | deepseek). No local LLMs."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o", temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-sonnet-5", temperature=0)
    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "deepseek-chat", temperature=0,
                          base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use openai | anthropic | deepseek)")


def audit(session_id: str, node: str, tool: str | None, content: str) -> None:
    """Append one row to agent_audit_log. Never lets a logging failure break the agent."""
    try:
        with get_engine().begin() as conn:
            conn.execute(
                text("INSERT INTO agent_audit_log (session_id, node_executed, tool_name, content) "
                     "VALUES (:s, :n, :t, :c)"),
                {"s": session_id, "n": node, "t": tool, "c": content},
            )
    except Exception as e:
        print(f"[audit] write failed: {e}")


def build_agent():
    llm = build_llm().bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    # TODO(observability): attach LangSmith tracing / DeepEval hooks here.
    def reasoner(state: AgentState, config):
        response = llm.invoke([SYSTEM_PROMPT] + state["messages"])
        session = config["configurable"]["thread_id"]
        calls = [f"{c['name']}({c['args']})" for c in response.tool_calls]
        audit(session, "reasoner", None, "; ".join(calls) if calls else str(response.content))
        return {"messages": [response]}

    def tools(state: AgentState, config):
        result = tool_node.invoke(state, config)
        session = config["configurable"]["thread_id"]
        for msg in result["messages"]:
            audit(session, "tools", msg.name, str(msg.content))
        return result

    graph = StateGraph(AgentState)
    graph.add_node("reasoner", reasoner)
    graph.add_node("tools", tools)
    graph.add_edge(START, "reasoner")
    graph.add_conditional_edges("reasoner", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "reasoner")
    return graph.compile(checkpointer=MemorySaver())


def ask(agent, question: str, thread_id: str) -> str:
    """Send one user message; returns the final answer text."""
    result = agent.invoke({"messages": [("user", question)]}, config={"configurable": {"thread_id": thread_id}})
    return result["messages"][-1].text


if __name__ == "__main__":
    agent = build_agent()
    session = f"cli-{uuid.uuid4().hex[:8]}"
    print("Cold-chain copilot ready. Type 'exit' to quit.")
    while (q := input("\nDispatcher > ").strip()).lower() not in {"exit", "quit"}:
        print(f"\n{ask(agent, q, session)}")
