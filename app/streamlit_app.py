import os
import streamlit as st
from pathlib import Path
import subprocess
import sys
import time

# ============================================================
#              AUTO-DETECT PROJECT ROOT
# ============================================================

def find_project_root(target="rag_pipeline.py"):
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / target).exists():
            return current
        current = current.parent
    raise FileNotFoundError(f"Cannot find project root containing {target}")

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline import (
    build_faiss_index,
    rag_query,
    PAGES_TEXT_PATH,
    OUTPUTS,
)

st.set_page_config(page_title="CERN Yellow Report RAG", layout="wide")

# ============================================================
# 1️⃣ Extraction + Index Build (same as before)
# ============================================================

st.sidebar.header("Setup")

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# PDF upload / selection
existing_pdfs = sorted(DATA_DIR.glob("*.pdf"))
pdf_choices = [p.name for p in existing_pdfs] if existing_pdfs else []
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
selected_name = st.sidebar.selectbox(
    "Or select a PDF from data/",
    options=["<none>"] + pdf_choices,
    index=0,
)

pdf_path = None
if uploaded_pdf is not None:
    target = DATA_DIR / uploaded_pdf.name
    with open(target, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    pdf_path = target
elif selected_name and selected_name != "<none>":
    pdf_path = DATA_DIR / selected_name

if pdf_path is None or not pdf_path.exists():
    st.sidebar.warning("Please upload or select a PDF.")
else:
    st.sidebar.success(f"Using PDF: {pdf_path.name}")

    # Run extraction only if pages_text.json is missing
    if not PAGES_TEXT_PATH.exists():
        st.sidebar.warning("Running extraction… this may take a while.")
        with st.spinner("Extracting data..."):
            env = os.environ.copy()
            env["PDF_PATH"] = str(pdf_path)
            p = subprocess.run(
                ["python", "extraction/pipeline.py"],
                cwd=PROJECT_ROOT,
                env=env
            )
            if p.returncode == 0:
                st.sidebar.success("Extraction complete.")
            else:
                st.sidebar.error("Extraction failed. Check logs.")
    else:
        st.sidebar.info("Extraction outputs found.")

    # Build or detect FAISS index
    if st.sidebar.button("Build / Rebuild FAISS Index"):
        with st.spinner("Building FAISS index..."):
            try:
                count = build_faiss_index()
                st.sidebar.success(f"Indexed {count} chunks.")
            except Exception as e:
                st.sidebar.error(f"Index build error: {e}")

    if (OUTPUTS / "faiss_index.bin").exists():
        st.sidebar.success("FAISS index ready.")
    else:
        st.sidebar.warning("No FAISS index found; build it once.")

# ============================================================
# 2️⃣ Chat Interface
# ============================================================

st.header("Chat with your PDF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Show answer and maybe sources
            st.markdown(msg["content"])
            # If there are sources in the message, display them
            for src in msg.get("sources", []):
                if src.get("type") == "figure":
                    img_path = Path(src["image_path"])
                    if not img_path.is_absolute():
                        img_path = PROJECT_ROOT / img_path
                    st.image(str(img_path), caption=src.get("caption", ""))
                elif src.get("type") == "table":
                    st.write(f"Table: Page {src.get('page')} (score {src.get('score')})")
                elif src.get("type") == "text":
                    st.write(f"Text: {src.get('text')[:200]}…")
        else:
            st.markdown(msg["content"])

# Chat input at the bottom
if prompt := st.chat_input("Ask something about the PDF…"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve answer from RAG
    if not (PAGES_TEXT_PATH.exists() and (OUTPUTS / "faiss_index.bin").exists()):
        answer = "Extraction or index missing. Please ensure you've run extraction and built the index."
        sources = []
    else:
        with st.spinner("Generating answer…"):
            result = rag_query(prompt, top_k=5)
            answer = result["answer"]
            sources = result["sources"]

    # Append assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    # Re-render to show the new messages
    # st.experimental_rerun()
