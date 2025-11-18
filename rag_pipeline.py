import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional LLM for answer generation
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# =====================================================================
#                           PATHS & CONSTANTS
# =====================================================================

BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"

PAGES_TEXT_PATH = OUTPUTS / "pages_text.json"
TABLES_INDEX_PATH = OUTPUTS / "tables_index.json"
FIGURES_INDEX_PATH = OUTPUTS / "figures_index.json"

FAISS_INDEX_PATH = OUTPUTS / "faiss_index.bin"
FAISS_META_PATH = OUTPUTS / "faiss_metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =====================================================================
#                      GLOBAL CACHES (MODELS, INDEX)
# =====================================================================

_embedding_model = None
_faiss_index = None
_faiss_metadata = None
_llm_pipe = None

# =====================================================================
#                        CHEMICAL TEXT DETECTION
# =====================================================================

CHEM_PATTERNS = [
    r"C\d+H\d+",              # C6H6, C2H4, etc.
    r"[A-Z][a-z]?\d+",        # H2O, CO2, etc.
    r"COOH|OH|NH2|CONH|CHO",  # functional groups
    r"-[A-Z][a-z]?-[A-Z]?",   # -C-O-, -N-H-, etc.
    r"[A-Za-z0-9]+\s*→\s*[A-Za-z0-9]+",  # reaction arrow
    r"poly\w+",               # polymer names: polyimide, polyamide, etc.
]


def detect_chemical_text(text: str) -> bool:
    text = text or ""
    for pat in CHEM_PATTERNS:
        if re.search(pat, text):
            return True
    return False


# =====================================================================
#                        LOADING UTILS
# =====================================================================

def load_pages():
    if not PAGES_TEXT_PATH.exists():
        return []
    with open(PAGES_TEXT_PATH, "r") as f:
        return json.load(f)


def load_tables():
    if not TABLES_INDEX_PATH.exists():
        return []
    with open(TABLES_INDEX_PATH, "r") as f:
        return json.load(f)


def load_figures():
    if not FIGURES_INDEX_PATH.exists():
        return []
    with open(FIGURES_INDEX_PATH, "r") as f:
        return json.load(f)


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_llm():
    """
    Lightweight text2text LLM for answer generation.
    Uses google/flan-t5-small (~300MB). If unavailable or fails,
    we gracefully degrade to extractive behavior.
    """
    global _llm_pipe
    if _llm_pipe is not None:
        return _llm_pipe

    model_name = "google/flan-t5-small"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _llm_pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1  # CPU
        )
    except Exception as e:
        print(f"[LLM] Warning: failed to load {model_name}: {e}")
        _llm_pipe = None

    return _llm_pipe


# =====================================================================
#                        TEXT CHUNKING
# =====================================================================

def chunk_text(text, max_tokens=220, stride=40):
    """
    Simple whitespace-based chunking. Not token-accurate, but good enough.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - stride

    return chunks


# =====================================================================
#                     BUILD / LOAD FAISS INDEX
# =====================================================================

def build_faiss_index():
    """
    Build FAISS index over text chunks from pages_text.json.
    Also stores metadata including chemical flags.
    """
    pages = load_pages()
    if not pages:
        raise RuntimeError(f"No pages found in {PAGES_TEXT_PATH}")

    model = get_embedding_model()

    texts = []
    meta = []

    print(f"[FAISS] Building chunks from {len(pages)} pages...")

    global_id = 0
    for page_obj in pages:
        page_num = page_obj.get("page")
        page_text = page_obj.get("text", "")
        if not page_text.strip():
            continue

        chunks = chunk_text(page_text, max_tokens=220, stride=40)
        for ch in chunks:
            if not ch.strip():
                continue

            chem_flag = 1 if detect_chemical_text(ch) else 0

            texts.append(ch)
            meta.append({
                "chunk_id": global_id,
                "page": page_num,
                "text": ch,
                "chem": chem_flag
            })
            global_id += 1

    print(f"[FAISS] Total chunks: {len(texts)}")

    # Embed
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    # Store
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[FAISS] Saved index to {FAISS_INDEX_PATH}")
    print(f"[FAISS] Saved metadata to {FAISS_META_PATH}")

    # Cache globals in-process
    global _faiss_index, _faiss_metadata
    _faiss_index = index
    _faiss_metadata = meta

    return len(texts)


def load_faiss_index():
    global _faiss_index, _faiss_metadata

    if _faiss_index is not None and _faiss_metadata is not None:
        return _faiss_index, _faiss_metadata

    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        raise RuntimeError("FAISS index or metadata missing. Run build_faiss_index() first.")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "r") as f:
        meta = json.load(f)

    _faiss_index = index
    _faiss_metadata = meta
    return _faiss_index, _faiss_metadata


# =====================================================================
#                       SEARCH: TEXT CHUNKS
# =====================================================================

def search_chunks(query: str, top_k: int = 5):
    """
    Returns top_k text chunks (no tables/figures) for context preview.
    Used by Streamlit for "Context Preview".
    """
    index, meta = load_faiss_index()
    model = get_embedding_model()

    q_emb = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(q_emb.reshape(1, -1), top_k)
    scores = D[0]
    idxs = I[0]

    results = []
    for score, idx in zip(scores, idxs):
        rec = meta[idx]
        results.append({
            "page": rec["page"],
            "text": rec["text"],
            "score": float(score),
        })

    return results


def search_text_chunks_full(query: str, top_k: int = 10):
    """
    Text chunk search that includes metadata (used inside rag_query).
    Adds a small boost for chemically rich chunks.
    """
    index, meta = load_faiss_index()
    model = get_embedding_model()

    q_emb = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(q_emb.reshape(1, -1), top_k * 2)  # more for re-ranking
    scores = D[0]
    idxs = I[0]

    results = []
    for score, idx in zip(scores, idxs):
        rec = meta[idx]

        # Boost if chunk has chemical info
        if rec.get("chem") == 1:
            score += 0.20

        results.append({
            "type": "text",
            "page": rec["page"],
            "text": rec["text"],
            "score": float(score),
            "chem": rec.get("chem", 0),
        })

    # Re-sort by boosted score and keep top_k
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results


# =====================================================================
#                       SEARCH: TABLES
# =====================================================================

def table_record_to_text(rec):
    page = rec.get("page", "?")
    preview = rec.get("preview", [])

    if not preview:
        return f"Table on page {page} (no preview rows)."

    header = preview[0]
    first_row = preview[1] if len(preview) > 1 else []

    header_str = " | ".join(h for h in header if h)
    row_str = " | ".join(r for r in first_row if r)

    if header_str and row_str:
        return (
            f"Table from page {page}. Columns: {header_str}. "
            f"Example row: {row_str}."
        )
    elif header_str:
        return f"Table from page {page}. Columns: {header_str}."
    else:
        return f"Table on page {page} (unable to infer columns)."


def search_tables(query: str, top_k: int = 3):
    tables = load_tables()
    if not tables:
        return []

    model = get_embedding_model()

    texts = []
    meta = []
    for rec in tables:
        summary = table_record_to_text(rec)
        texts.append(summary)
        meta.append({
            "page": rec.get("page", None),
            "table_csv": rec.get("table_csv", ""),
            "summary": summary,
            "preview": rec.get("preview", [])
        })

    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False
    )

    q_emb = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = embs @ q_emb
    idxs = np.argsort(-scores)[:top_k]

    results = []
    for i in idxs:
        m = meta[i]
        results.append({
            "type": "table",
            "score": float(scores[i]),
            "page": m["page"],
            "table_csv": m["table_csv"],
            "summary": m["summary"],
            "preview": m["preview"],
        })

    return results


# =====================================================================
#                       SEARCH: FIGURES / GRAPHS
# =====================================================================

def figure_to_text(fig):
    caption = fig.get("caption", "")
    page = fig.get("page", None)
    kind = fig.get("kind", "image")
    return f"{kind} on page {page}: {caption}"


def search_figures(query: str, top_k: int = 3):
    figs = load_figures()
    if not figs:
        return []

    model = get_embedding_model()

    texts = []
    meta = []

    for f in figs:
        summary = figure_to_text(f)
        texts.append(summary)
        meta.append({
            "page": f.get("page", None),
            "image_path": f.get("image_path", ""),
            "caption": f.get("caption", ""),
            "kind": f.get("kind", "image"),
        })

    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False
    )

    q_emb = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = embs @ q_emb
    idxs = np.argsort(-scores)[:top_k]

    results = []
    for i in idxs:
        m = meta[i]
        results.append({
            "type": "figure",
            "score": float(scores[i]),
            "page": m["page"],
            "image_path": m["image_path"],
            "caption": m["caption"],
            "kind": m["kind"],
        })

    return results


# =====================================================================
#                      RAG QUERY: MAIN ENTRY POINT
# =====================================================================

def build_context_text(text_hits, table_hits, fig_hits, max_chars=3500):
    """
    Build a context string for the LLM, mixing text, tables, and figure captions.
    """
    parts = []

    if text_hits:
        parts.append("TEXT EXTRACTS:\n")
        for h in text_hits:
            parts.append(f"[Page {h['page']}] {h['text']}\n")

    if table_hits:
        parts.append("\nTABLE SUMMARIES:\n")
        for t in table_hits:
            parts.append(f"[Page {t['page']}] {t['summary']}\n")

    if fig_hits:
        parts.append("\nFIGURE CAPTIONS:\n")
        for f in fig_hits:
            parts.append(f"[Page {f['page']}] {f['caption']}\n")

    ctx = "\n".join(parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n... [truncated]"
    return ctx


def is_chem_query(query: str) -> bool:
    query = query.lower()
    chem_keywords = [
        "chemical", "formula", "bond", "structure", "polymer",
        "functional group", "epoxy", "polyimide", "polyamide",
        "resin", "backbone", "molecule"
    ]
    return any(k in query for k in chem_keywords)


def rag_query(query: str, top_k: int = 5):
    """
    Main RAG function:
    - retrieves text chunks, tables, and figures
    - builds a context string
    - passes to a small LLM if available
    - returns {'answer': str, 'sources': [...]}
    """
    text_hits = search_text_chunks_full(query, top_k=top_k)
    table_hits = search_tables(query, top_k=3)
    fig_hits = search_figures(query, top_k=3)

    # Determine if chemical-aware hint is needed
    chem_context_present = any(h.get("chem") == 1 for h in text_hits)
    chem_query_flag = is_chem_query(query)
    use_chem_hint = chem_context_present or chem_query_flag

    chem_hint = ""
    if use_chem_hint:
        chem_hint = (
            "The user is asking about chemical formulas, diagrams, or structures.\n"
            "Explain clearly:\n"
            "- Identify key functional groups and bond types (amide, aromatic, ether, etc.).\n"
            "- Describe the polymer backbone or molecular structure.\n"
            "- If radiation behavior or material properties are mentioned, summarize the relationship.\n"
            "- If no image is available, do NOT mention images, just explain the chemistry in words.\n\n"
        )

    context_text = build_context_text(text_hits, table_hits, fig_hits, max_chars=3500)

    llm = get_llm()
    if llm is not None:
        prompt = (
            "You are a scientific assistant helping to interpret a CERN materials report.\n"
            "Use ONLY the provided context to answer, and be concise but precise.\n\n"
            f"{chem_hint}"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:\n"
        )
        try:
            out = llm(
                prompt,
                max_new_tokens=256,
                do_sample=False
            )
            answer = out[0]["generated_text"].strip()
        except Exception as e:
            print(f"[LLM] Error during generation: {e}")
            # fallback: concatenated context
            answer = (
                "LLM generation failed; here is the most relevant context:\n\n"
                + context_text[:1200]
            )
    else:
        # No LLM available -> simple extractive fallback
        answer = (
            "LLM model is not available. Here is the most relevant extracted context:\n\n"
            + context_text[:1200]
        )

    # Build a unified sources list
    sources = []

    for h in text_hits:
        sources.append({
            "type": "text",
            "page": h["page"],
            "score": h["score"],
            "text": h["text"],
            "chem": h.get("chem", 0),
        })

    for t in table_hits:
        sources.append({
            "type": "table",
            "page": t["page"],
            "score": t["score"],
            "summary": t["summary"],
            "preview": t["preview"],
            "table_csv": t["table_csv"],
        })

    for f in fig_hits:
        sources.append({
            "type": "figure",
            "page": f["page"],
            "score": f["score"],
            "image_path": f["image_path"],
            "caption": f["caption"],
            "kind": f["kind"],
        })

    # Sort all sources by score desc
    sources = sorted(sources, key=lambda s: s.get("score", 0.0), reverse=True)

    return {
        "answer": answer,
        "sources": sources
    }


# =====================================================================
#                      CLI TEST (OPTIONAL)
# =====================================================================

if __name__ == "__main__":
    print("[RAG] Testing pipeline...")
    if not FAISS_INDEX_PATH.exists():
        print("[RAG] FAISS index not found. Building now...")
        build_faiss_index()

    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        res = rag_query(q, top_k=5)
        print("\n=== ANSWER ===")
        print(res["answer"])
        print("\n=== TOP SOURCES ===")
        for s in res["sources"][:3]:
            print(f"- {s['type']} page {s.get('page')}, score={s.get('score'):.3f}")
