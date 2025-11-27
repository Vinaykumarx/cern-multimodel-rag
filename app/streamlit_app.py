# app/streamlit_app.py

import sys
import os
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Ensure project root is on sys.path so we can import core.*
# -------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # cern-multimodel-rag/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag_pipeline import RAGPipeline  # noqa: E402


@st.cache_resource
def get_pipeline() -> Optional[RAGPipeline]:
    load_dotenv()

    db_uri = os.getenv("LANCEDB_URI", "lancedb")  # local folder by default
    table_name = os.getenv("LANCEDB_TABLE", "cern_demo")

    pipe = RAGPipeline(
        db_uri=db_uri,
        table_name=table_name,
        enable_blip=True,
        device="cpu",
    )
    return pipe


def run():
    st.set_page_config(page_title="CERN Multimodal RAG", layout="wide")

    st.title("🔬 CERN Multimodal RAG (PDF + Images + LanceDB)")
    st.write(
        "Upload a CERN PDF or use the preset file, ingest it into LanceDB, "
        "and run semantic Q&A over the contents."
    )

    pipeline = get_pipeline()
    if pipeline is None:
        st.stop()

    # Sidebar: ingestion controls
    st.sidebar.header("Ingestion")

    default_pdf_path = os.environ.get(
        "DEFAULT_PDF_PATH",
        "data/CERN_Yellow_Report_357576.pdf",
    )
    use_default = st.sidebar.checkbox(
        f"Use default PDF ({default_pdf_path})",
        value=True,
    )

    uploaded_file = st.sidebar.file_uploader("Or upload a PDF", type=["pdf"])
    ingest_button = st.sidebar.button("Ingest into LanceDB")

    if ingest_button:
        if use_default and os.path.exists(default_pdf_path):
            pdf_path = default_pdf_path
        elif uploaded_file is not None:
            tmp_dir = "uploads"
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_path = tmp_path
        else:
            st.sidebar.error("No PDF selected or uploaded.")
            st.stop()

        with st.spinner(f"Ingesting {pdf_path} into LanceDB…"):
            pipeline.ingest_pdf(pdf_path)
        st.sidebar.success("Ingestion completed.")

    st.markdown("---")

    # Q&A section
    st.subheader("Ask a question about the ingested PDFs")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input(
        "Your question:",
        placeholder="e.g., What does the report say about radiation damage?",
    )
    top_k = st.slider("Top-K retrieved chunks", min_value=3, max_value=15, value=5, step=1)

    if st.button("Search") and user_query.strip():
        with st.spinner("Retrieving from LanceDB…"):
            hits = pipeline.query(user_query, top_k=top_k)

        st.session_state.chat_history.append(
            {
                "user": user_query,
                "hits": hits,
            }
        )

    # Render chat history (newest first)
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {entry['user']}")
        st.markdown("**Top retrieved chunks:**")
        for h in entry["hits"]:
            st.markdown(
                f"- (score={h['score']:.3f}, page={h.get('page')})\n\n"
                f"  `{h.get('source', '')}`\n\n"
                f"  > {h['text'][:600]}..."
            )
        st.markdown("---")


if __name__ == "__main__":
    run()
