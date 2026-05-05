"""Streamlit UI for the A2A resume-roaster project."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from agents.orchestrator.main import run_orchestrator


def _extract_text_from_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as error:
            raise RuntimeError("pypdf is required for PDF upload support. Install with `pip install pypdf`.") from error

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        reader = PdfReader(tmp_path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return text

    raise RuntimeError("Unsupported file type. Upload .txt or .pdf")


def main() -> None:
    st.set_page_config(page_title="Resume Roaster", page_icon="🔥", layout="wide")
    st.title("Resume Roaster - Multi-Agent A2A UI")

    st.markdown(
        "This UI sends your resume to Engineer and HR A2A agents in parallel through the orchestrator."
    )

    col1, col2 = st.columns(2)
    with col1:
        engineer_url = st.text_input("Engineer Agent URL", value="http://127.0.0.1:8102")
    with col2:
        hr_url = st.text_input("HR Agent URL", value="http://127.0.0.1:8103")

    upload = st.file_uploader("Upload resume (.txt or .pdf)", type=["txt", "pdf"])
    manual_text = st.text_area("Or paste resume text", height=220)

    if st.button("Run Roast", type="primary"):
        try:
            resume_text = ""
            if upload is not None:
                resume_text = _extract_text_from_upload(upload).strip()
            elif manual_text.strip():
                resume_text = manual_text.strip()

            if not resume_text:
                st.error("Provide resume input via upload or text area.")
                return

            with st.spinner("Running Engineer + HR agents..."):
                result = run_orchestrator(
                    resume_text=resume_text,
                    engineer_base_url=engineer_url,
                    hr_base_url=hr_url,
                )

            final_roast = result.get("final_roast", {})
            st.subheader("Final Verdict")
            st.metric("Overall Score", final_roast.get("overall_score_out_of_10", "N/A"))
            st.write("Decision:", final_roast.get("verdict", "N/A"))

            st.subheader("Engineer Lens")
            st.write("Score:", final_roast.get("engineer_score_out_of_10", "N/A"))
            for item in final_roast.get("top_criticisms", {}).get("engineer", []):
                st.write(f"- {item}")
            st.write("Strength:", final_roast.get("strengths", {}).get("engineer", ""))

            st.subheader("HR Lens")
            st.write("Score:", final_roast.get("hr_score_out_of_10", "N/A"))
            for item in final_roast.get("top_criticisms", {}).get("hr", []):
                st.write(f"- {item}")
            st.write("Strength:", final_roast.get("strengths", {}).get("hr", ""))

            with st.expander("Raw JSON"):
                st.json(result)

        except Exception as error:
            st.error(f"Run failed: {error}")


if __name__ == "__main__":
    main()
