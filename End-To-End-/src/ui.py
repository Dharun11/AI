"""Streamlit dispatcher console. Stateless UI: all reasoning happens in the orchestrator."""
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.orchestrator import ask, build_agent  # noqa: E402

st.set_page_config(page_title="Cold-Chain Copilot", page_icon="🧊")
st.title("🧊 Cold-Chain Logistics Copilot")


@st.cache_resource
def get_agent():
    return build_agent()


if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

if question := st.chat_input("Ask about fleet telemetry, weather or SOP compliance..."):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"), st.spinner("Checking telemetry, weather and SOPs..."):
        answer = ask(get_agent(), question, st.session_state.thread_id)
        st.markdown(answer)
    st.session_state.history += [("user", question), ("assistant", answer)]
