"""
Campus Companion – Interactive RAG Learning Lab
Main Streamlit application entry point.
"""

import streamlit as st
from components import (
    sidebar,
    dashboard,
    knowledge_base,
    chat,
    pipeline_explorer,
    experiments,
    learning_centre,
    settings_about,
)

st.set_page_config(
    page_title="Campus Companion – Interactive RAG Learning Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Dark Navy / Charcoal + Purple & Electric Blue SaaS Theme
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top right, #161b33 0%, #0b0f19 55%);
            color: #f1f5f9;
        }
        [data-testid="stSidebar"] {
            background-color: #0e1322;
            border-right: 1px solid #1e293b;
        }
        .cc-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.72), rgba(15, 23, 42, 0.88));
            border: 1px solid rgba(139, 92, 246, 0.25);
            border-radius: 12px;
            padding: 16px 18px;
        }
        .cc-card-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .cc-muted {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }
        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(56, 189, 248, 0.2);
            padding: 12px 14px;
            border-radius: 10px;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #7c3aed, #2563eb);
            border: none;
            color: #ffffff;
            font-weight: 600;
        }
        .stButton > button[kind="primary"]:hover {
            filter: brightness(1.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    selected_page = sidebar.render_sidebar()

    if selected_page == "🏠 Dashboard":
        dashboard.render_dashboard()
    elif selected_page == "📚 Knowledge Base":
        knowledge_base.render_knowledge_base()
    elif selected_page == "💬 Ask Campus Companion":
        chat.render_chat()
    elif selected_page == "🔬 RAG Pipeline Explorer":
        pipeline_explorer.render_pipeline_explorer()
    elif selected_page == "🧪 RAG Experiments":
        experiments.render_experiments()
    elif selected_page == "🎓 Learning Centre":
        learning_centre.render_learning_centre()
    elif selected_page == "⚙️ Settings & About":
        settings_about.render_settings_about()


if __name__ == "__main__":
    main()

