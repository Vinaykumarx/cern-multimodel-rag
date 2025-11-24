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



cern-multimodel-rag/
├── app/
│ └── streamlit_app.py # ChatGPT-like frontend
├── extraction/
│ ├── extract_text.py
│ ├── extract_images.py
│ ├── extract_tables.py
│ ├── extract_graphs.py
│ ├── caption_images.py
│ ├── build_metadata.py
│ └── pipeline.py # orchestrates extraction
├── rag_pipeline.py # Qdrant Cloud + embeddings
├── data/
│ └── CERN_Report.pdf # default PDF (optional)
├── outputs/
│ └── .gitkeep # extraction results stored here
├── docker-compose.yml # Docling + MinIO + Streamlit
├── Dockerfile # Streamlit backend container
├── requirements.txt
└── README.md

> Note: The `data/` folder is intentionally **empty** in the repository.
> You must provide your own PDFs (any name is fine).

---
# 🔬 CERN Multimodal RAG System
A full end-to-end, multimodal Retrieval-Augmented Generation (RAG) system for analyzing CERN Yellow Reports using:

- **Docling OCR** (Docker)
- **MinIO (S3 storage)**
- **Qdrant Cloud** (vector database)
- **Sentence-Transformers** (embeddings)
- **Streamlit Chat UI** (ChatGPT-like)
- **BLIP captions** for figures/graphs

This system extracts text, tables, images, figures & graphs from a PDF and enables semantic question-answering with sources.

---

## 📁 Project Structure

