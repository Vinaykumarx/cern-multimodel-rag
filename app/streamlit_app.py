import streamlit as st
import subprocess
import os
import time
from rag_pipeline import rag_query

st.set_page_config(
    page_title="CERN RAG Demo",
    page_icon="🔬",
    layout="wide"
)

# ---------------------------
# UI HEADER
# ---------------------------
st.title("🔬 CERN Yellow Report RAG")
st.caption("Ask questions about the CERN PDF. Powered by Docling + Qdrant Cloud.")

# Chat session
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------
# PDF INGESTION SECTION
# ---------------------------
st.header("📄 PDF Ingestion (Optional)")

uploaded_pdf = st.file_uploader("Upload a PDF to re-index", type=["pdf"])

if uploaded_pdf:
    st.info("Processing PDF inside Docker…")

    # Save PDF to container data folder
    pdf_path = "/app/data/uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    # Trigger ingestion
    with st.spinner("Running ingestion... (Docling → Qdrant)"):
        p = subprocess.run(
            ["python", "scripts/ingest_pdf.py"],
            cwd="/app"
        )

    if p.returncode == 0:
        st.success("PDF successfully indexed!")
    else:
        st.error("Ingestion failed. Check logs.")


# ---------------------------
# CHAT SECTION
# ---------------------------
st.header("💬 Ask a Question")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Enter a question")
    else:
        with st.spinner("Retrieving + generating answer…"):
            result = rag_query(query)

        st.session_state.messages.append(
            {"q": query, "a": result["answer"], "sources": result["sources"]}
        )


# ---------------------------
# DISPLAY CHAT HISTORY
# ---------------------------
for msg in st.session_state.messages:
    st.markdown(f"**🧑 You:** {msg['q']}")
    st.markdown(f"**🤖 RAG:** {msg['a']}")
    st.markdown("---")

    with st.expander("Sources"):
        for s in msg["sources"]:
            st.write(f"- Score: {s['score']}")
            st.write(f"- Pages: {s['page_numbers']}")
            st.write(f"```{s['text'][:500]}```")
            st.write("---")
