# import pdfplumber, json
# from pathlib import Path
# BASE = Path(__file__).resolve().parent.parent

# out_dir = BASE/'outputs'
# out_dir.mkdir(exist_ok=True)

# tables_index = []

# with pdfplumber.open(str(pdf_path)) as pdf:
#     for i, page in enumerate(pdf.pages):
#         try:
#             tables = page.extract_tables()
#             for t_idx, table in enumerate(tables):
#                 csv_path = out_dir/f"page_{i+1}_table_{t_idx+1}.csv"
#                 with open(csv_path,'w') as cf:
#                     for row in table:
#                         cf.write(",".join(['' if c is None else str(c).replace("\n"," ") for c in row]) + "\n")
#                 tables_index.append({'page': i+1, 'table_csv': str(csv_path)})
#         except Exception as e:
#             print("Error on page", i+1, e)

# with open(out_dir/'tables_index.json','w') as f:
#     json.dump(tables_index, f, indent=2)

# print("Tables saved to", out_dir)

import os
import pdfplumber
import json
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
pdf_path = BASE / "data" / "CERN_Yellow_Report_357576.pdf"
out_dir = BASE / "outputs"
out_dir.mkdir(exist_ok=True)

tables_index = []

PDF_ENV = os.getenv('PDF_PATH')
if PDF_ENV:
    pdf_path = Path(PDF_ENV)
else:
    pdf_path = BASE / 'data' / 'CERN_Yellow_Report_357576.pdf'

with pdfplumber.open(str(pdf_path)) as pdf:
    for i, page in enumerate(pdf.pages):
        try:
            tables = page.extract_tables()
        except Exception as e:
            print(f"[TABLES] Error extracting tables on page {i+1}: {e}")
            continue

        for t_idx, table in enumerate(tables):
            if not table:
                continue

            csv_path = out_dir / f"page_{i+1}_table_{t_idx+1}.csv"
            preview_rows = []

            with open(csv_path, "w", newline="") as cf:
                writer = csv.writer(cf)
                for row in table:
                    clean = [
                        "" if c is None else str(c).replace("\n", " ").strip()
                        for c in row
                    ]
                    writer.writerow(clean)
                    if len(preview_rows) < 5:
                        preview_rows.append(clean)

            tables_index.append({
                "page": i + 1,
                "table_csv": str(csv_path),
                "preview": preview_rows
            })

with open(out_dir / "tables_index.json", "w") as f:
    json.dump(tables_index, f, indent=2)

print(f"[TABLES] Saved {len(tables_index)} tables to {out_dir}")