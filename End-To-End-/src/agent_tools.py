"""Agent tools: telemetry SQL (read-only view), live weather, SOP retrieval (Pinecone)."""
import os
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MAX_ROWS = 10


@lru_cache
def get_engine():
    """Engine for the least-privilege agent account (SELECT on the view, INSERT on the audit log)."""
    return create_engine(URL.create(
        "mysql+pymysql",
        username=os.getenv("SQL_AGENT_USER", "usr_fde_ro"),
        password=os.getenv("SQL_AGENT_PASSWORD", ""),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "supply_chain"),
    ))


@lru_cache
def get_retriever():
    # Lazy: loading the local embedding model is slow, so only pay for it on first SOP search.
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore

    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    store = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX", "cold-chain-sop"), embedding=embeddings)
    return store.as_retriever(search_kwargs={"k": 3})


@tool
def query_telemetry_db(sql_query: str) -> str:
    """Run a single MySQL SELECT against the view vw_active_fleet.
    Columns: Timestamp, Latitude, Longitude, Current_Temperature_C, Cargo_Condition_Code,
    Risk_Classification, Delay_Probability, Port_Congestion_Level, Route_Risk_Index.
    Use MySQL syntax (LIMIT, not TOP)."""
    query = sql_query.strip().rstrip(";")
    # The DB role is the real barrier; this guard just gives the model a clear error.
    if not query.lower().startswith("select") or ";" in query:
        return "SECURITY BLOCK: only a single SELECT statement is allowed."
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchmany(MAX_ROWS)
            if not rows:
                return "No records matched the query."
            lines = [f"COLUMNS: {', '.join(result.keys())}"] + [str(tuple(r)) for r in rows]
            return "\n".join(lines)
    except Exception as e:
        return f"Database error: {e}"


@tool
def fetch_corridor_conditions(latitude: float, longitude: float) -> str:
    """Get live weather (temperature, wind speed) at a GPS point from the Open-Meteo API."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": latitude, "longitude": longitude, "current_weather": "true"},
            timeout=6,
        )
        resp.raise_for_status()
        w = resp.json()["current_weather"]
        return (f"Live weather at ({latitude}, {longitude}): "
                f"{w['temperature']}°C, wind {w['windspeed']} km/h, weather code {w['weathercode']}.")
    except Exception as e:
        return f"Weather API unavailable ({e}). Report this data as missing; do not guess."


@tool
def search_compliance_sop(query: str) -> str:
    """Search the company SOPs in the Pinecone index: temperature thresholds, rerouting rules, escalation tiers."""
    try:
        docs = get_retriever().invoke(query)
        if not docs:
            return "No matching SOP clauses found."
        return "\n\n".join(
            f"[Source: {d.metadata.get('source_file') or d.metadata.get('source', 'SOP')}]\n{d.page_content}"
            for d in docs
        )
    except Exception as e:
        return f"SOP retrieval unavailable ({e}). Report this data as missing; do not guess."


ALL_TOOLS = [query_telemetry_db, fetch_corridor_conditions, search_compliance_sop]


if __name__ == "__main__":
    print(query_telemetry_db.invoke("SELECT Latitude, Longitude, Current_Temperature_C FROM vw_active_fleet LIMIT 2"))
    print(fetch_corridor_conditions.invoke({"latitude": 33.77, "longitude": -118.19}))
    print(search_compliance_sop.invoke("temperature rules for fresh perishables"))
