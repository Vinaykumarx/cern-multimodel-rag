# scripts/test_pdf_extract.py

import os
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pdf_loader import extract_text_pages, extract_page_images  # noqa: E402


def main():
    pdf_path = os.getenv(
        "TEST_PDF",
        str(ROOT / "data" / "CERN_Yellow_Report_357576.pdf"),
    )
    print(f"[Test] PDF path: {pdf_path}")

    if not os.path.exists(pdf_path):
        print("[ERROR] PDF does not exist.")
        return

    pages = extract_text_pages(pdf_path)
    print(f"[Text] Extracted {len(pages)} pages.")
    for p in pages[:3]:
        print(f"Page {p['page']}:")
        print(p["text"][:300].replace("\n", " "))
        print("----")

    images = extract_page_images(pdf_path)
    print(f"[Images] Extracted {len(images)} images.")
    if images:
        first = images[0]
        print(
            f"First image on page {first['page']}, type={type(first['image'])}, "
            f"size={first['image'].size}"
        )


if __name__ == "__main__":
    main()
