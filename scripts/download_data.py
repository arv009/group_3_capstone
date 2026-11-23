"""Download the Home Credit Default Risk competition data into data/raw/.

Requires a Kaggle API token (~/.kaggle/access_token or ~/.kaggle/kaggle.json)
and that you have accepted the competition rules on kaggle.com.

Usage:  python scripts/download_data.py
"""
from pathlib import Path

import kagglehub

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

path = kagglehub.competition_download("home-credit-default-risk", output_dir=str(OUT_DIR))

print("Path to competition files:", path)
for f in sorted(Path(path).iterdir()):
    if f.is_file():
        print(f"  {f.name:40s} {f.stat().st_size / 1e6:8.1f} MB")
