

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from agents.orchestrator.main import run_orchestrator

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _extract_text_from_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as error:
            raise RuntimeError(
                "pypdf is required for PDF support. Run: pip install pypdf"
            ) from error
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        reader = PdfReader(tmp_path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    raise RuntimeError("Unsupported file type. Upload .txt or .pdf")


def _score_color(score: float) -> str:
    if score >= 7.5:
        return "#22c55e"   # green
    if score >= 5.5:
        return "#f59e0b"   # amber
    return "#ef4444"       # red


def _verdict_badge(verdict: str) -> str:
    if verdict == "interview":
        return "🟢 INTERVIEW"
    return "🔴 REJECT"


def _render_score_bar(label: str, score: float, color: str) -> None:
    pct = int((score / 10) * 100)
    st.markdown(f"**{label}** &nbsp; `{score}/10`")
    st.markdown(
        f"""
        <div style="background:#e5e7eb;border-radius:8px;height:14px;width:100%;margin-bottom:10px;">
          <div style="background:{color};width:{pct}%;height:14px;border-radius:8px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_criticisms(items: List[str], icon: str = "❌") -> None:
    for item in items:
        st.markdown(
            f"""
            <div style="background:#fef2f2;border-left:4px solid #ef4444;
                        padding:10px 14px;border-radius:6px;margin-bottom:8px;font-size:0.92rem;">
              {icon} {item}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_strength(text: str, icon: str = "✅") -> None:
    st.markdown(
        f"""
        <div style="background:#f0fdf4;border-left:4px solid #22c55e;
                    padding:10px 14px;border-radius:6px;margin-bottom:8px;font-size:0.92rem;">
          {icon} {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_tool_signals(signals: Dict[str, Any], agent_label: str) -> None:
    if not signals:
        return
    with st.expander(f"🔬 {agent_label} — Tool Signal Details"):
        for key, data in signals.items():
            if not isinstance(data, dict):
                continue
            score = data.get("score", "—")
            summary = data.get("summary", "")
            label = key.replace("_", " ").title()
            st.markdown(f"**{label}** — score `{score}/10`")
            if summary:
                st.caption(summary)
            detail_keys = [k for k in data if k not in {"score", "summary"}]
            if detail_keys:
                detail = {k: data[k] for k in detail_keys}
                st.json(detail)


# ─────────────────────────────────────────────
# Sidebar — Architecture overview
# ─────────────────────────────────────────────

def _render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/fire-element.png",
            width=60,
        )
        st.markdown("## Resume Roaster")
        st.caption("Multi-Agent A2A Resume Reviewer")
        st.divider()

        st.markdown("### How it works")
        st.markdown(
            """
            1. 📝 You submit a resume
            2. 🔀 Orchestrator fans out to **2 agents in parallel**
            3. 🤖 **Engineer Agent** *(LangChain ReAct)*  
               Runs 4 technical tools → synthesizes verdict
            4. 👩‍💼 **HR Agent** *(Semantic Kernel ReAct)*  
               Runs 4 HR tools → synthesizes verdict
            5. 🔗 Orchestrator merges → final score + decision
            """
        )
        st.divider()

        st.markdown("### Agent Servers")
        engineer_url = st.text_input(
            "Engineer Agent URL", value="http://127.0.0.1:8102"
        )
        hr_url = st.text_input(
            "HR Agent URL", value="http://127.0.0.1:8103"
        )
        st.divider()

        st.markdown("### Tech Stack")
        st.markdown(
            """
            `LangChain` · `Semantic Kernel`  
            `FastAPI` · `Azure OpenAI / OpenAI`  
            `A2A Protocol` · `Pydantic` · `httpx`
            """
        )

    return engineer_url, hr_url


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Resume Roaster 🔥",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    engineer_url, hr_url = _render_sidebar()

    # ── Header ──────────────────────────────
    st.markdown(
        """
        <h1 style="font-size:2.4rem;margin-bottom:0;">🔥 Resume Roaster</h1>
        <p style="color:#6b7280;font-size:1.05rem;margin-top:4px;">
          Two AI agents. One brutal verdict. Powered by A2A protocol.
        </p>
        <hr style="margin:16px 0 24px 0;">
        """,
        unsafe_allow_html=True,
    )

    # ── Architecture banner ──────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 1, 2, 1, 2])
    with c1:
        st.markdown(
            """
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                        padding:14px;text-align:center;">
              <div style="font-size:1.6rem;">📝</div>
              <div style="font-weight:600;margin-top:4px;">Resume Input</div>
              <div style="font-size:0.78rem;color:#6b7280;">paste · upload .txt/.pdf</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div style='text-align:center;font-size:1.8rem;padding-top:24px;'>⚡</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div style="background:#faf5ff;border:1px solid #e9d5ff;border-radius:10px;
                        padding:14px;text-align:center;">
              <div style="font-size:1.6rem;">🔀</div>
              <div style="font-weight:600;margin-top:4px;">Orchestrator</div>
              <div style="font-size:0.78rem;color:#6b7280;">parallel A2A fan-out</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            "<div style='text-align:center;font-size:1.8rem;padding-top:24px;'>⚡</div>",
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            """
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                        padding:14px;text-align:center;">
              <div style="font-size:1.6rem;">🏆</div>
              <div style="font-weight:600;margin-top:4px;">Final Verdict</div>
              <div style="font-size:0.78rem;color:#6b7280;">score · decision · roast</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input ────────────────────────────────
    st.markdown("### 📄 Resume Input")
    tab_paste, tab_upload = st.tabs(["Paste Text", "Upload File"])
    with tab_paste:
        manual_text = st.text_area(
            "Paste resume text here",
            height=240,
            placeholder="John Doe\nSoftware Engineer\n5 years experience in Python, AWS, Kubernetes...",
            label_visibility="collapsed",
        )
    with tab_upload:
        upload = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔥 Run Roast", type="primary", use_container_width=True)

    if not run_btn:
        return

    # ── Collect resume text ──────────────────
    resume_text = ""
    try:
        if upload is not None:
            resume_text = _extract_text_from_upload(upload).strip()
        elif manual_text.strip():
            resume_text = manual_text.strip()
    except Exception as error:
        st.error(f"Could not read resume: {error}")
        return

    if not resume_text:
        st.warning("Please paste or upload a resume before running.")
        return

    # ── Run pipeline ─────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Running Multi-Agent Pipeline")
    step1 = st.empty()
    step2 = st.empty()
    step3 = st.empty()
    step4 = st.empty()

    step1.info("📡 Sending resume to Engineer Agent (port 8102) and HR Agent (port 8103) in parallel...")

    try:
        with st.spinner("Agents are reasoning with their tools. This takes 10–30 seconds..."):
            result = run_orchestrator(
                resume_text=resume_text,
                engineer_base_url=engineer_url,
                hr_base_url=hr_url,
            )
    except Exception as error:
        st.error(f"Pipeline failed: {error}")
        st.caption("Make sure both agent servers are running before clicking Run Roast.")
        return

    step1.success("✅ Both A2A agents responded.")
    step2.success("✅ Engineer artifact received.")
    step3.success("✅ HR artifact received.")
    step4.success("✅ Orchestrator merged both verdicts.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Final verdict banner ─────────────────
    final_roast = result.get("final_roast", {})
    overall = float(final_roast.get("overall_score_out_of_10", 0))
    verdict = final_roast.get("verdict", "reject")
    eng_score = float(final_roast.get("engineer_score_out_of_10", 0))
    hr_score = float(final_roast.get("hr_score_out_of_10", 0))

    badge_color = "#22c55e" if verdict == "interview" else "#ef4444"
    badge_text = "🟢 INTERVIEW — Move this candidate forward." if verdict == "interview" else "🔴 REJECT — Not ready for an interview slot."

    st.markdown(
        f"""
        <div style="background:{badge_color}18;border:2px solid {badge_color};
                    border-radius:14px;padding:20px 28px;margin-bottom:24px;">
          <div style="font-size:1.6rem;font-weight:700;color:{badge_color};">
            {badge_text}
          </div>
          <div style="font-size:1.1rem;color:#374151;margin-top:8px;">
            Overall Blended Score: <strong>{overall}/10</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Engineer: <strong>{eng_score}/10</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            HR: <strong>{hr_score}/10</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Score bars ───────────────────────────
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        _render_score_bar("Overall Score", overall, _score_color(overall))
    with bc2:
        _render_score_bar("🤖 Engineer Score", eng_score, _score_color(eng_score))
    with bc3:
        _render_score_bar("👩‍💼 HR Score", hr_score, _score_color(hr_score))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two agent panels ─────────────────────
    st.markdown("### 🔍 Agent Analysis Breakdown")
    left, right = st.columns(2)

    eng_criticisms = final_roast.get("top_criticisms", {}).get("engineer", [])
    eng_strength = final_roast.get("strengths", {}).get("engineer", "")
    hr_criticisms = final_roast.get("top_criticisms", {}).get("hr", [])
    hr_strength = final_roast.get("strengths", {}).get("hr", "")

    with left:
        st.markdown(
            """
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                        padding:14px 18px;margin-bottom:14px;">
              <span style="font-size:1.15rem;font-weight:700;">🤖 Engineer Agent</span><br>
              <span style="font-size:0.8rem;color:#6b7280;">LangChain · ReAct · 4 Technical Tools</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Criticisms**")
        _render_criticisms(eng_criticisms, "⚠️")
        st.markdown("**Genuine Strength**")
        _render_strength(eng_strength, "💡")

        # tool signals
        eng_artifact = result.get("engineer_artifact", {})
        eng_data = (eng_artifact.get("parts") or [{}])[0].get("data", {})
        _render_tool_signals(eng_data.get("tool_signals", {}), "Engineer Tools")

    with right:
        st.markdown(
            """
            <div style="background:#f5f3ff;border:1px solid #ddd6fe;border-radius:10px;
                        padding:14px 18px;margin-bottom:14px;">
              <span style="font-size:1.15rem;font-weight:700;">👩‍💼 HR Agent</span><br>
              <span style="font-size:0.8rem;color:#6b7280;">Semantic Kernel · ReAct · 4 HR Tools</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Criticisms**")
        _render_criticisms(hr_criticisms, "⚠️")
        st.markdown("**Genuine Strength**")
        _render_strength(hr_strength, "💡")

        # tool signals
        hr_artifact = result.get("hr_artifact", {})
        hr_data = (hr_artifact.get("parts") or [{}])[0].get("data", {})
        _render_tool_signals(hr_data.get("tool_signals", {}), "HR Tools")

    # ── Raw JSON ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🗂️ Full Raw JSON Response"):
        st.json(result)

    # ── Footer ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;color:#9ca3af;font-size:0.82rem;padding:12px 0 4px 0;">
          Built with LangChain · Semantic Kernel · FastAPI · A2A Protocol · Streamlit
          &nbsp;|&nbsp; Azure OpenAI / OpenAI
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

