# CERN Multimodal RAG System

End-to-end **PDF ingestion + multimodal extraction + Retrieval-Augmented Generation (RAG)** system
for CERN Yellow Reports and radiation-materials datasets (MaxRAD / Imhotep).

The pipeline:
- extracts **text, tables, figures/graphs, and captions** from scientific PDFs,
- builds a **FAISS + Sentence-Transformers** semantic index over text,
- links text chunks to **images and tables** via captions,
- exposes an interactive **Streamlit QA app** where you can ask questions and request figures/tables.

> This repository is designed to run both **locally** and on **Google Colab** with GPU.

---

## Features

- 🔍 **PDF ingestion + parsing**
  - Uses `PyMuPDF` / `fitz`, `camelot` / `tabula`, and OpenCV-style heuristics.
  - Extracts:
    - raw page text
    - table structures
    - figure/graph crops
    - image captions and local context

- 🧠 **Semantic RAG**
  - Sentence-Transformers encoder (all-MiniLM or similar).
  - FAISS index on normalized embeddings.
  - Simple re-ranking and metadata-rich retrieval results.

- 🖼 **Multimodal grounding**
  - Text chunks are linked to:
    - table metadata + CSV content
    - image / graph crops
    - captions around each figure
  - QA answers can surface both relevant text and visuals.

- 💻 **Streamlit demo app**
  - Upload or select a PDF from the `data/` directory.
  - Run the full extraction pipeline from the UI.
  - Build / rebuild FAISS index.
  - Ask questions like:
    - *"Show me graphs related to tensile strength under irradiation"*
    - *"Which table summarizes material composition for alloy X?"*

---

## Repository structure

```text
.
├── app/
│   └── streamlit_app.py        # Main Streamlit UI
├── extraction/
│   ├── extract_text.py         # Page text extraction
│   ├── extract_tables.py       # Table extraction (Camelot/Tabula)
│   ├── extract_images.py       # Figure / image crops
│   ├── extract_graphs.py       # Graph-like region detection
│   ├── caption_images.py       # Local caption / context extraction
│   └── pipeline.py             # Orchestrates Stage-1 extraction
├── llm/
│   ├── __init__.py
│   └── qa.py                   # Lightweight LLM-style answer generation
├── rag_pipeline.py             # FAISS index build + retrieval pipeline
├── data/                       # (Empty) place your PDFs here
├── outputs/                    # Extraction outputs (JSON, PNGs, etc.)
├── notebook/
│   └── Stage1_Demo.ipynb       # Dev notebook for Stage-1
├── requirements.txt
├── .gitignore
└── README.md
```

> Note: The `data/` folder is intentionally **empty** in the repository.
> You must provide your own PDFs (any name is fine).

---

## Installation (local)

```bash
git clone https://github.com/YOUR_USERNAME/cern-multimodal-rag.git
cd cern-multimodal-rag

python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Then run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Usage

1. **Add a PDF**
   - Put one or more PDFs into the `data/` folder  
     *(e.g., `data/yr_2023_materials.pdf`)*  
   - Or use the **Upload** widget inside the Streamlit app.

2. **Start the app**

   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **In the Web UI**
   - Select or upload a PDF.
   - The app will:
     - check if extraction outputs exist,
     - if missing, run `extraction/pipeline.py` **using the selected PDF**, and
     - store outputs in `outputs/`.

   - Then:
     - click **"Build FAISS Index"** to index the extracted text.
     - ask your question in the **Ask a Question** box.

   - The answer view:
     - shows generated text,
     - lists retrieved text chunks,
     - and, when relevant, surfaces associated tables / figures.

---

## Running in Google Colab (GPU) – Suggested Pattern

1. Create a Colab notebook and add:

   ```python
   !git clone https://github.com/YOUR_USERNAME/cern-multimodal-rag.git
   %cd cern-multimodal-rag

   !pip install -r requirements.txt
   ```

2. (Optional) Use `pyngrok` or `cloudflared` to expose the Streamlit app:

   ```python
   !pip install pyngrok

   from pyngrok import ngrok
   public_url = ngrok.connect(8501)
   print("Public URL:", public_url)

   !streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
   ```

3. Upload PDFs via the Streamlit UI or copy them into `data/`.

---

## Design notes

- The extraction scripts now respect a `PDF_PATH` environment variable.
  - If `PDF_PATH` is set, that file is used.
  - Otherwise, they default to `data/CERN_Yellow_Report_357576.pdf` for backwards compatibility.
- The Streamlit app always:
  - lets you upload/select **any** PDF name,
  - passes the chosen path to the extraction pipeline via `PDF_PATH`.

This means the system is **no longer tied to a specific file name**:
any valid `.pdf` in `data/` (or uploaded) will work.

---

## Roadmap / Future Work

- Better figure/graph detection and caption association.
- Richer table understanding (cell types, units, uncertainties).
- Stronger LLM backend for answer generation.
- HuggingFace Spaces or other persistent hosting for the demo.

---

## License

This project is intended for research and educational purposes around
scientific-document understanding and RAG systems.
