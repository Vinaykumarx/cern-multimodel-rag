import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
out = BASE/'outputs'

data = {}

for fname, key in [
    ('pages_text.json', 'pages'),
    ('tables_index.json', 'tables'),
    ('figures_index.json', 'figures')
]:
    fpath = out/fname
    if fpath.exists():
        with open(fpath) as f:
            data[key] = json.load(f)
    else:
        data[key] = []

with open(BASE/'metadata.json','w') as f:
    json.dump(data, f, indent=2)

print("Saved metadata.json")
