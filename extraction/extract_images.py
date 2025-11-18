import fitz
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent.parent
PDF_ENV = os.getenv('PDF_PATH')
if PDF_ENV:
    pdf_path = Path(PDF_ENV)
else:
    pdf_path = BASE / 'data' / 'CERN_Yellow_Report_357576.pdf'
out_dir = BASE/'outputs'
images_dir = out_dir/'images'
images_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(str(pdf_path))
count = 0

for i, page in enumerate(doc):
    for img in page.get_images(full=True):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        out = images_dir/f"page_{i+1}_img_{xref}.png"

        if pix.n < 5:
            pix.save(out)
        else:
            pix0 = fitz.Pixmap(fitz.csRGB, pix)
            pix0.save(out)
            pix0 = None

        pix = None
        count += 1

print("Saved", count, "images to", images_dir)