# CERN Multimodal RAG (Local LanceDB Version)

This project is a **multimodal RAG prototype** for CERN PDFs:

- Extracts **text** and **images** from PDFs using PyMuPDF
- Uses **BLIP** to caption figures
- Chunks text + figure captions
- Embeds chunks with **SentenceTransformer** (`all-MiniLM-L6-v2`)
- Stores vectors locally in **LanceDB**
- Exposes a **Streamlit** UI to ingest PDFs and run semantic search

## Project Structure

```text
cern-multimodel-rag/
├── app/
│   ├── streamlit_app.py
│   ├── __init__.py
│   └── ui_components/
│       └── __init__.py
├── core/
│   ├── config.py
│   ├── pdf_loader.py
│   ├── image_captioner.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store_lance.py
│   └── rag_pipeline.py
├── data/
│   └── CERN_Yellow_Report_357576.pdf
├── uploads/
├── lancedb/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/
│   ├── test_lancedb.py
│   ├── test_pdf_extract.py
│   ├── test_embeddings.py
│   ├── clean_env.sh
│   └── debug_env.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
