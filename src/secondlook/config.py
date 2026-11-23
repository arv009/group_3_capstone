"""Project-wide paths and constants, so every notebook and app page agrees on them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_ZIP = RAW_DIR / "home-credit-default-risk.zip"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"  # 1 = client had payment difficulties (our default label)
RANDOM_STATE = 42

# Kept in the feature table for the fairness audit, but NEVER used as model inputs.
PROTECTED_COLS = ["CODE_GENDER", "AGE_YEARS"]
# Kept in the feature table for auditing, but never model inputs (decided in notebook 08):
# family status and number of children are protected traits in several jurisdictions, and
# EMPLOYED_AGE_RATIO (years employed / age) lets a model recover age exactly from YEARS_EMPLOYED.
EXCLUDED_INPUTS = ["NAME_FAMILY_STATUS", "CNT_CHILDREN", "EMPLOYED_AGE_RATIO"]

# Fixed applicant-level split. The calibration slice is reserved for probability
# calibration and conformal prediction, so it is never used for training.
SPLIT_FRACTIONS = {"train": 0.60, "valid": 0.15, "calib": 0.10, "test": 0.15}

# Applicants sampled for the demo app, so it stays fast (Capstone Guide, section 7).
DEMO_SAMPLE_SIZE = 50_000
