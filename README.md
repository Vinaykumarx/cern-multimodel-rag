# 🌐 CERN Multimodal RAG System  
### Hybrid PDF Extraction + Semantic Search + ChatGPT-style Q&A for CERN Yellow Reports & Radiation-Materials Data

This repository implements a **full multimodal Retrieval-Augmented Generation (RAG) pipeline** for scientific document understanding.  
It is designed specifically for CERN materials-science datasets, including:

- CERN Yellow Reports  
- MaxRAD radiation-materials database  
- Imhotep radiation test data  
- CDS PDF documents  

The system extracts **text, tables, figures, graphs, and captions**, semantically indexes them using **FAISS**, and answers scientific questions using a **lightweight LLM** with chemical-aware reasoning.

---

# 🚀 Features

### ✅ **Multimodal PDF Extraction**
- Text extraction (PyMuPDF)  
- Table extraction (pdfplumber → CSV)  
- Graph detection (OpenCV heuristic cropping)  
- Image extraction & captioning (BLIP-base)  
- Chemical-text recognition (polymer chains, functional groups)

### ✅ **Semantic Indexing (FAISS + MiniLM)**
- Efficient chunking  
- Sentence-transformer embeddings  
- Chemical-aware ranking & boosting logic  
- Supports page-level, table-level, and figure-level retrieval

### ✅ **LLM-powered Scientific QA**
- Lightweight LLM (Flan-T5-small by default)  
- Chemistry-aware prompts  
- Summarizes chemical formulas & polymer structures  
- Describes graphs and tables intelligently  
- Stable even on limited hardware

### ✅ **Streamlit User Interface**
- ChatGPT-like interface  
- Multimodal context preview (text + table snippets + figure captions)  
- GPU-friendly hosting via Google Colab + Cloudflare Tunnel

---

# 📦 Repository Structure

```text
cern-multimodal-rag/
│
├── app/
│   └── streamlit_app.py         # Streamlit UI
│
├── extraction/                  # Stage-1 PDF ingestion pipeline
│   ├── extract_text.py
│   ├── extract_tables.py
│   ├── extract_images.py
│   ├── caption_images.py
│   ├── build_metadata.py
│   └── pipeline.py
│
├── data/                        # Place PDFs here (kept out of Git)
│
├── outputs/                     # Auto-generated metadata/index files
│
├── rag_pipeline.py              # RAG engine + FAISS + LLM
│
├── requirements.txt             # Dependencies
│
├── CERN_RAG_Colab.ipynb         # Colab notebook for GPU hosting
│
└── README.md                    # This file
# cern-multimodel-rag
