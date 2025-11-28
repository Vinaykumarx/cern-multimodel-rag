# app/streamlit_app.py

import sys
import os
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Disable tokenizers parallel warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Groq client
from groq import Groq

# -------------------------------------------------------------------
# Ensure project root is on sys.path so we can import core.*
# -------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag_pipeline import RAGPipeline


# -------------------------------------------------------------------
# Initialize pipeline
# -------------------------------------------------------------------
@st.cache_resource
def get_pipeline() -> Optional[RAGPipeline]:
    load_dotenv()

    db_uri = os.getenv("LANCEDB_URI", "lancedb")
    table_name = os.getenv("LANCEDB_TABLE", "cern_demo")

    return RAGPipeline(
        db_uri=db_uri,
        table_name=table_name,
        enable_blip=True,
        device="cpu",
    )


def run():
    st.set_page_config(page_title="CERN Multimodal RAG", layout="wide")

    st.title("🔬 CERN Multimodal RAG (PDF + Figures + LanceDB + Groq LLM)")
    st.caption("PDF ingestion → BLIP figure extraction → LanceDB → Hybrid RAG → Groq summarisation")

    pipeline = get_pipeline()
    if pipeline is None:
        st.stop()

    # -------------------------------------------------------------------
    # 📥 SIDEBAR INGESTION CONTROLS
    # -------------------------------------------------------------------
    st.sidebar.header("Ingestion")

    default_pdf = "data/CERN_Yellow_Report_357576.pdf"
    use_default = st.sidebar.checkbox("Use default PDF", value=True)

    uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if st.sidebar.button("Ingest PDF"):
        if use_default:
            pdf_path = default_pdf
        else:
            if uploaded is None:
                st.sidebar.error("Upload a PDF file first.")
                st.stop()
            os.makedirs("uploads", exist_ok=True)
            pdf_path = os.path.join("uploads", uploaded.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded.getbuffer())

        with st.spinner("Ingesting PDF…"):
            pipeline.ingest_pdf(pdf_path)

        st.sidebar.success("PDF ingested successfully!")

    # Multi-PDF ingestion
    st.sidebar.markdown("---")
    folder = st.sidebar.text_input("Folder ingestion", "sample_reports")

    if st.sidebar.button("Ingest Folder"):
        if not os.path.exists(folder):
            st.sidebar.error("Folder does not exist.")
        else:
            with st.spinner("Ingesting folder…"):
                pipeline.ingest_folder(folder)
            st.sidebar.success("Folder ingestion completed!")

    st.markdown("---")

    # -------------------------------------------------------------------
    # 🔎 QUESTION INPUT + RAG SEARCH
    # -------------------------------------------------------------------
    st.subheader("Ask a question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input(
        "Your question:",
        placeholder="e.g., Summarize all figures related to radiation damage"
    )
    top_k = st.slider("Top-K results", 3, 15, 5)

    if st.button("Search"):
        if not query.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving chunks…"):
                hits = pipeline.query(query, top_k=top_k)
            st.session_state.chat_history.append({"query": query, "hits": hits})

    # -------------------------------------------------------------------
    # 🧠 GROQ LLM SUMMARISATION
    # -------------------------------------------------------------------
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]

        if st.button("🧠 Generate LLM Summary"):
            prompt = pipeline.build_summary_prompt(last["query"], last["hits"])

            st.markdown("#### Prompt sent to Groq")
            st.code(prompt)

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                st.error("GROQ_API_KEY is missing in your .env")
                st.stop()

            client = Groq(api_key=api_key)

            # Main + fallback model logic
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # NEW recommended best model
                    messages=[
                        {"role": "system", "content": "You summarise scientific RAG content accurately."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )

            except Exception:
                st.warning(
                    "⚠️ `llama-3.3-70b-versatile` failed or is unavailable. "
                    "Using fallback model: `llama-3.3-8b-instant`."
                )

                response = client.chat.completions.create(
                    model="llama-3.3-8b-instant",
                    messages=[
                        {"role": "system", "content": "You summarise scientific RAG content accurately."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )

            # Extract content (Groq format)
            answer = response.choices[0].message.content

            st.markdown("### 🧠 Final Answer")
            st.write(answer)

    st.markdown("---")

    # -------------------------------------------------------------------
    # 📚 RETRIEVAL HISTORY (FIGURE PREVIEWS)
    # -------------------------------------------------------------------
    st.subheader("📚 Retrieval History")

    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"### **Query:** {entry['query']}")

        for hit in entry["hits"]:
            label = hit["label"]
            score = hit["score"]
            page = hit["page"]

            with st.expander(f"{label} — Page {page}  | Score={score:.3f}"):

                st.markdown("#### Extracted Text")
                st.write(hit["text"][:1000] + "...")

                if hit.get("figure_captions"):
                    st.markdown("#### Figure Captions")
                    for cap in hit["figure_captions"]:
                        st.write(f"- {cap}")

                if hit.get("figure_paths"):
                    st.markdown("#### Extracted Figures")
                    cols = st.columns(3)
                    for i, p in enumerate(hit["figure_paths"]):
                        try:
                            im = Image.open(p)
                            cols[i % 3].image(im, caption=os.path.basename(p), use_container_width=True)
                        except Exception:
                            cols[i % 3].error(f"Could not load {p}")

        st.markdown("---")


if __name__ == "__main__":
    run()
