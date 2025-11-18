import fitz, json
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent.parent
PDF_ENV = os.getenv('PDF_PATH')
if PDF_ENV:
    pdf_path = Path(PDF_ENV)
else:
    pdf_path = BASE / 'data' / 'CERN_Yellow_Report_357576.pdf'
out_dir = BASE/'outputs'
out_dir.mkdir(exist_ok=True)

doc = fitz.open(str(pdf_path))
pages = []

for i, page in enumerate(doc):
    pages.append({'page': i+1, 'text': page.get_text('text')})

with open(out_dir/'pages_text.json','w') as f:
    json.dump(pages, f, indent=2)

print("Extracted text to", out_dir/'pages_text.json')