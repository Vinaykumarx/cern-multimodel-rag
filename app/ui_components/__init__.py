# app/ui_components/__init__.py

import streamlit as st


def render_header():
    st.title("🔬 CERN Multimodal RAG")
    st.caption("PDF + Figures + LanceDB + SentenceTransformer")


def render_footer():
    st.markdown("---")
    st.caption("Prototype CERN multimodal RAG – local LanceDB backend.")
