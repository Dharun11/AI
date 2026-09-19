"""Streamlit viewer for the Agentic RAG ingestion API.

A pure HTTP client of the FastAPI service in src/api - upload documents,
trigger /ingest with a chosen configuration, and inspect the chunks +
metadata that run produced. Only shows the just-triggered run's results; it
does not browse previously-ingested data.

Run alongside the API:
    uvicorn api.main:app --reload
    streamlit run src/streamlit_app/app.py
"""

import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # AGENTIC_RAG_API_URL, if set, lives in .env like everything else here

DEFAULT_API_BASE_URL = os.environ.get("AGENTIC_RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic RAG Ingestion", layout="wide")
st.title("Agentic RAG — Ingestion Viewer")


@st.cache_data(ttl=60)
def fetch_strategies(base_url: str) -> dict[str, list[str]]:
    response = requests.get(f"{base_url}/strategies", timeout=10)
    response.raise_for_status()
    return response.json()


def render_chunk_detail(chunk: dict) -> None:
    if chunk["content_type"] == "table":
        st.markdown(chunk["text"])
    else:
        st.code(chunk["text"], language=None)
    st.json(chunk["metadata"])
    st.caption(
        f"chunk_id={chunk['chunk_id']} · embedding_model={chunk['embedding_model']} "
        f"· embedding_dim={chunk['embedding_dim']} · content_hash={chunk['content_hash'][:16]}…"
    )


def render_document_result(doc: dict, key_prefix: str) -> None:
    with st.expander(f"{doc['source_filename']}  (doc_id={doc['doc_id'] or '—'})", expanded=True):
        if doc["error"]:
            st.error(f"Failed at stage '{doc['error']['stage']}': {doc['error']['message']}")
            return

        if not doc["chunks"]:
            st.info("No chunks produced for this document.")
            return

        rows = [
            {
                "chunk_index": c["metadata"]["chunk_index"],
                "content_type": c["content_type"],
                "section_path": " > ".join(c["metadata"]["section_path"]) or "(root)",
                "token_count": c["token_count"],
                "embedded": c["embedding_model"] is not None,
                "text_preview": c["text"][:150].replace("\n", " "),
            }
            for c in doc["chunks"]
        ]

        selection_key = f"{key_prefix}_table"
        state = st.dataframe(
            rows,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=selection_key,
        )

        selected_rows = getattr(getattr(state, "selection", None), "rows", []) or []
        if selected_rows:
            render_chunk_detail(doc["chunks"][selected_rows[0]])
        else:
            st.caption("Select a row above to see the full chunk text and metadata.")


with st.sidebar:
    st.header("Connection")
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)

    try:
        strategies = fetch_strategies(api_base_url)
        api_reachable = True
    except requests.RequestException as exc:
        st.error(f"Could not reach API: {exc}")
        strategies = {"parsers": [], "chunkers": [], "embedders": [], "stores": []}
        api_reachable = False

    st.header("Project")
    project_id = st.text_input("Project ID", value="default")

    st.header("Parsing & chunking")
    parser_name = st.selectbox("Parser", strategies["parsers"] or ["docling"])
    chunking_strategy = st.selectbox("Chunking strategy", strategies["chunkers"] or ["character_v1"])
    chunk_size_tokens = st.number_input("Chunk size (tokens)", min_value=1, value=512)
    chunk_overlap_tokens = st.number_input("Chunk overlap (tokens)", min_value=0, value=50)

    st.header("Embedder")
    embedder_name = st.selectbox("Embedder", strategies["embedders"] or ["sentence_transformer"])
    embedder_model_name = st.text_input("Model name", value="BAAI/bge-small-en-v1.5")
    embedder_batch_size = st.number_input("Batch size", min_value=1, value=32)

    st.header("Store")
    store_name = st.selectbox("Store", strategies["stores"] or ["in_memory", "qdrant", "pinecone"])
    store_config: dict = {"name": store_name}
    if store_name == "qdrant":
        store_config["collection_name"] = st.text_input("Collection name", value="chunks")
        store_config["path"] = st.text_input("Local storage path (optional)", value="")
    elif store_name == "pinecone":
        store_config["pinecone_index_name"] = st.text_input("Index name (optional, defaults to project id)", value="")
        store_config["pinecone_cloud"] = st.text_input("Cloud", value="aws")
        store_config["pinecone_region"] = st.text_input("Region", value="us-east-1")

st.subheader("Upload documents")
uploaded_files = st.file_uploader("Documents to ingest", accept_multiple_files=True)
run_clicked = st.button("Ingest", disabled=not api_reachable or not uploaded_files)

if run_clicked:
    config = {
        "project_id": project_id,
        "parser_name": parser_name,
        "chunking": {
            "strategy": chunking_strategy,
            "chunk_size_tokens": chunk_size_tokens,
            "chunk_overlap_tokens": chunk_overlap_tokens,
        },
        "embedder": {
            "name": embedder_name,
            "model_name": embedder_model_name,
            "batch_size": embedder_batch_size,
        },
        "store": {k: v for k, v in store_config.items() if v not in (None, "")},
    }
    files = [("files", (f.name, f.getvalue())) for f in uploaded_files]

    with st.spinner("Ingesting…"):
        try:
            response = requests.post(
                f"{api_base_url}/ingest",
                files=files,
                data={"config": json.dumps(config)},
                timeout=600,
            )
            response.raise_for_status()
            st.session_state["last_result"] = response.json()
        except requests.RequestException as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            st.error(f"Ingest failed: {detail}")

result = st.session_state.get("last_result")
if result:
    st.subheader(f"Results — project '{result['project_id']}'")
    counters = result["counters"]
    cols = st.columns(len(counters))
    for col, (name, value) in zip(cols, counters.items()):
        col.metric(name, value)

    for i, doc in enumerate(result["documents"]):
        render_document_result(doc, key_prefix=f"doc_{i}")
