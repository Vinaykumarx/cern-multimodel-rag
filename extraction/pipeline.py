import os
import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent

steps = [
    "extract_text.py",
    "extract_images.py",
    "extract_tables.py",
    "extract_graphs.py",
    "caption_images.py",
    "build_metadata.py",
]

def run_step(script):
    print(f"[Pipeline] Running: {script}")
    p = subprocess.run(
        [sys.executable, str(BASE / script)],
        env=os.environ.copy()
    )
    if p.returncode != 0:
        print(f"❌ Step failed: {script}")
        sys.exit(1)
    print(f"[Pipeline] Completed: {script}")


if __name__ == "__main__":
    print("========== CERN EXTRACTION PIPELINE ==========")
    for s in steps:
        run_step(s)

    print("========== PIPELINE COMPLETED ==========")
