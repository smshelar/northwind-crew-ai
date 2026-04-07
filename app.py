"""
app.py
-------
Streamlit frontend for the Northwind Agentic AI project.

Run with:
    streamlit run app.py
"""

import sys
import os


# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import DB setup function
from northwind_db.db_manager import download_db
# Auto-download database if missing
download_db()


# Streamlit + Orchestrator
import streamlit as st
from crew.northwind_crew import NorthwindCrew

# ---------------- Page config ---------------- #
st.set_page_config(
    page_title="Northwind Agentic AI",
    layout="wide"
)

# load CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/styles.css")



# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Northwind Agentic AI")
    st.caption("Powered by CrewAI · Ask a business question!")
    st.markdown("---")
    for q in [
        "What were the top 5 products by sales?",
        "Which customers have the highest order frequency?",
        "Show me the sales trend for the last quarter.",
    ]:
        if st.button(q, use_container_width=True):
            st.session_state["user_query"] = q

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Ask the Northwind AI")
st.caption("Query your Northwind database using plain English")

col_input, col_btn = st.columns([5, 1])
with col_input:
    user_query = st.text_input(
        "",
        value=st.session_state.get("user_query", ""),
        placeholder="e.g., What were the top 5 products by sales?",
        key="user_query",
        label_visibility="collapsed",
    )
with col_btn:
    run = st.button("Ask the AI ✦", type="primary", use_container_width=True)

if run and user_query.strip():
    if "crew" not in st.session_state:
        st.session_state["crew"] = NorthwindCrew()

    with st.spinner("🤖 Agents at work..."):
        result = st.session_state["crew"].run(user_query)

    with st.expander("Pipeline steps", expanded=False):
        for step in result["steps"]:
            st.markdown(f"- {step}")

    if not result["success"]:
        if "no results" in str(result["error"]).lower() or "no data" in str(result["error"]).lower():
            st.warning("🔍 No data found. Note: The Northwind database only contains data from **1996, 1997, and 1998**.")
        else:
            st.warning("😕 Something went wrong. Try rephrasing.")
        with st.expander("Technical details"):
            st.code(result["error"])
        st.stop()
        
    if result["sql_query"]:
        with st.expander("Generated SQL", expanded=True):
            st.code(result["sql_query"], language="sql")

    if result.get("figure"):
        st.subheader("📊 Visualization")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.pyplot(result["figure"], use_container_width=True)
        with col2:
            st.markdown("### 💡 Insight")
            st.success(result["insight"] or "No insight generated.")

    if result.get("summary"):
        st.subheader("🧠 Summary")
        st.info(result["summary"])

    if result["dataframe"] is not None:
        st.subheader(f"Data ({len(result['dataframe'])} rows)")
        st.dataframe(result["dataframe"], use_container_width=True)