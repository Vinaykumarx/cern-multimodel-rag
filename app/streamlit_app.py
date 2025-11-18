import os
import streamlit as st
from pathlib import Path
import subprocess
import sys
import time

if "COLAB_GPU" in os.environ:
    IS_COLAB = True
else:
    IS_COLAB = False

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


# ============================================================
#                    STREAMLIT UI CONFIG
# ============================================================

st.set_page_config(page_title="CERN Yellow Report RAG", layout="wide")

st.title("🔬 CERN Yellow Report — RAG QA Demo")
st.write("Ask technical questions or request figures/graphs from the CERN Yellow Report.")


# ============================================================
#           1️⃣ EXTRACTION CHECK + AUTO-RUN
# ============================================================

st.header("1️⃣ Extraction Status")

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Let user either upload a PDF or pick one from data/
existing_pdfs = sorted(DATA_DIR.glob("*.pdf"))
default_label = None
if existing_pdfs:
    default_label = str(existing_pdfs[0].name)
    pdf_choices = [p.name for p in existing_pdfs]
else:
    pdf_choices = []

uploaded_pdf = st.file_uploader("Upload a PDF (optional)", type=["pdf"])
selected_name = None
if pdf_choices:
    selected_name = st.selectbox(
        "Or select a PDF from data/ (optional)",
        options=["<none>"] + pdf_choices,
        index=0
    )
pdf_path = None
if uploaded_pdf is not None:
    # Save uploaded file into data/
    target = DATA_DIR / uploaded_pdf.name
    with open(target, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    pdf_path = target
elif selected_name and selected_name != "<none>":
    pdf_path = DATA_DIR / selected_name

st.write("PDF Path:", pdf_path if pdf_path is not None else "(no PDF selected)")

# Check PDF exists
if pdf_path is None or not pdf_path.exists():
    st.warning("Please upload a PDF or place one into data/ and select it above.")
else:
    st.success("✅ PDF selected and found.")
    st.warning("❗ Missing extracted data. Running extraction pipeline automatically...")
    with st.spinner("Running extraction pipeline..."):
        env = os.environ.copy()
        if pdf_path is not None:
            env["PDF_PATH"] = str(pdf_path)
        p = subprocess.run(
            ["python", "extraction/pipeline.py"],
            cwd=PROJECT_ROOT,
            env=env
        )
        if p.returncode == 0:
            st.success("Extraction completed automatically.")
        else:
            st.error("❌ Extraction failed. Check logs in the terminal.")
else:
    st.success("Extracted text found (pages_text.json).")


st.divider()


# ============================================================
#           2️⃣ BUILD / REBUILD FAISS INDEX
# ============================================================

st.header("2️⃣ Build / Rebuild FAISS Index")

if st.button("Build FAISS Index"):
    with st.spinner("Building FAISS index (10–20s, depending on CPU)..."):
        try:
            count = build_faiss_index()
            st.success(f"Indexed {count} semantic text chunks.")
        except Exception as e:
            st.error(f"Error while building index: {e}")

if (OUTPUTS / "faiss_index.bin").exists():
    st.success("✅ FAISS index detected.")
else:
    st.warning("FAISS index missing. Please build it at least once.")


st.divider()


# ============================================================
#               3️⃣ CHAT / QUESTION ANSWERING
# ============================================================

st.header("3️⃣ Ask a Question")

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.text_area(
    "Enter your question (e.g., 'show me a graph with some caption in this pdf'):",
    height=90
)
top_k = st.slider("Top-k retrieved items:", 1, 10, 5)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        status = st.empty()
        status.info("🔍 Retrieving relevant information and generating answer...")

        t0 = time.time()
        result = rag_query(query, top_k)
        t1 = time.time()

        st.session_state.chat.append({
            "q": query,
            "a": result["answer"],
            "sources": result["sources"],
            "time": round(t1 - t0, 2)
        })

        status.success(f"✅ Answer generated in {round(t1 - t0, 2)}s")


# ============================================================
#               4️⃣ DISPLAY CHAT HISTORY
# ============================================================
st.header("Conversation")

if st.session_state.chat:
    # 1) Show plain chat history
    for item in st.session_state.chat:
        st.markdown(f"**🧑‍💻 You:** {item['q']}")
        st.markdown(f"**🤖 RAG:** {item['a']}")
        st.markdown("---")

    # 2) Show sources ONLY for the latest answer
    last = st.session_state.chat[-1]
    sources = last["sources"]

    has_figure = any(s.get("type") == "figure" or s.get("image_path") for s in sources)
    has_table = any(s.get("type") == "table" for s in sources)
    has_text = any(s.get("type") == "text" for s in sources)

    st.subheader("Sources for the latest answer")

    # ---- Figures ----
    if has_figure:
        st.markdown("**🖼 Relevant Figures:**")
        for s in sources:
            if s.get("type") == "figure" or s.get("image_path"):
                img_path = s.get("image_path", None)
                caption = s.get("caption", "")
                score = s.get("score", 0.0)
                page = s.get("page", None)

                if img_path:
                    img_path_obj = Path(img_path)
                    if not img_path_obj.is_absolute():
                        img_path_obj = PROJECT_ROOT / img_path_obj
                    st.image(
                        str(img_path_obj),
                        caption=f"{caption}\n(Page: {page}, Score: {score:.3f})"
                    )
                else:
                    st.write(f"Caption: {caption} (Score: {score:.3f})")

                st.write("---")

    # ---- Tables ----
    if has_table:
        st.markdown("**📊 Relevant Tables:**")
        for s in sources:
            if s.get("type") == "table":
                page = s.get("page", "?")
                score = s.get("score", 0.0)
                summary = s.get("summary", "")
                preview = s.get("preview", [])
                table_csv = s.get("table_csv", "")

                st.write(f"- **Page {page}**, score {score:.3f}")
                st.write(f"Summary: {summary}")

                # Show a small preview as a table if available
                if preview:
                    try:
                        import pandas as pd
                        df = pd.DataFrame(preview[1:], columns=preview[0])
                        st.table(df)
                    except Exception:
                        # Fallback: just show raw rows
                        st.write("Preview rows:")
                        for row in preview[:5]:
                            st.write(" | ".join(row))

                if table_csv:
                    st.caption(f"CSV file: {table_csv}")

                st.write("---")

    # ---- Text-only sources (if no figures/tables) ----
    if has_text and not (has_figure or has_table):
        with st.expander("Sources (text chunks)"):
            for s in sources:
                if s.get("type") == "text":
                    page = s.get("page", "?")
                    score = s.get("score", 0.0)
                    text = s.get("text", "")
                    st.write(f"- **Page {page}**, score {score:.3f}")
                    st.write(f"`{text[:300]}`")
                    st.write("---")

    st.caption(f"⏱ Latest answer generated in {last['time']} seconds")
else:
    st.info("No questions asked yet.")