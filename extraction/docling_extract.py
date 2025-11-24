import requests
import json
import base64
from pathlib import Path

DOCLING_URL = "http://docling-serve:5001/v1/chunk/hybrid/file"

# Folders where we store processed output
OUTPUTS_DIR = Path("outputs")
IMAGES_DIR = OUTPUTS_DIR / "images"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

def ensure_dirs():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def decode_and_save_image(image_base64: str, filename: str, target_folder: Path) -> str:
    """Decodes base64 → saves image to file → returns file path"""
    try:
        binary = base64.b64decode(image_base64)
        out_path = target_folder / filename
        with open(out_path, "wb") as f:
            f.write(binary)
        return str(out_path)
    except Exception as e:
        print(f"[WARN] Failed decoding image {filename}: {e}")
        return ""


def call_docling(pdf_path: Path):
    """Sends PDF to Docling Serve and returns the raw JSON response."""
    pdf_bytes = pdf_path.read_bytes()

    files = {
        "files": ("document.pdf", pdf_bytes, "application/pdf")
    }

    # Minimal, stable options recommended for hybrid chunker.
    data = {
        "include_converted_doc": "false",
        "target_type": "inbody",
        "convert_from_formats": "pdf",
        "convert_do_ocr": "true",
        "convert_force_ocr": "false",
        "convert_pipeline": "legacy",
        "convert_do_table_structure": "true",
        "convert_include_images": "true",
        "chunking_use_markdown_tables": "false",
        "chunking_include_raw_text": "false",
        "chunking_merge_peers": "true"
    }

    print(f"[Docling] Sending PDF to {DOCLING_URL} ...")

    response = requests.post(
        DOCLING_URL,
        files=files,
        data=data,
        timeout=900
    )
    response.raise_for_status()

    print(f"[Docling] Extraction completed. (status={response.status_code})")
    return response.json()


def process_docling_output(docling_json: dict):
    """Extracts chunks and images/tables/figures from Docling output."""

    ensure_dirs()

    # Save raw chunks.json
    chunks_path = OUTPUTS_DIR / "chunks.json"
    chunks_path.write_text(json.dumps(docling_json, indent=2))
    print(f"[SAVE] chunks.json saved → {chunks_path}")

    # Process documents (images, tables, figures)
    documents = docling_json.get("documents", [])

    for doc in documents:
        content = doc.get("content", {})
        json_doc = content.get("json_content", {})

        if not json_doc:
            continue

        # Pictures / figures
        for pic in json_doc.get("pictures", []):
            filename = pic.get("filename", "img.png")
            image_b64 = pic.get("binary_data", "")
            if image_b64:
                decode_and_save_image(image_b64, filename, FIGURES_DIR)

        # Tables
        for table in json_doc.get("tables", []):
            filename = table.get("filename", "table.png")
            image_b64 = table.get("binary_data", "")
            if image_b64:
                decode_and_save_image(image_b64, filename, TABLES_DIR)

        # Images (general)
        for img in json_doc.get("images", []):
            filename = img.get("filename", "image.png")
            image_b64 = img.get("binary_data", "")
            if image_b64:
                decode_and_save_image(image_b64, filename, IMAGES_DIR)

    print("[Docling] Images, tables, and figures processed.")


def extract_pdf(pdf_path: Path):
    """Main function used by pipeline.py"""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[Extract] Processing PDF → {pdf_path.name}")

    docling_json = call_docling(pdf_path)
    process_docling_output(docling_json)

    return docling_json
