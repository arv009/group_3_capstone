"""Project-wide paths and constants, so every notebook and app page agrees on them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_ZIP = DATA_DIR / "raw" / "home-credit-default-risk.zip"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = INTERIM_DIR / "secondlook.db"

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
RANDOM_STATE = 42

# Applicants sampled for the demo app, so it stays fast (Capstone Guide, section 7).
DEMO_SAMPLE_SIZE = 50_000
