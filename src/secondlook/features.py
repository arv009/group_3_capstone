"""Build the model-ready feature table: cleaned application data + SQL history summaries.

Run from the project root to rebuild everything from the Kaggle zip:

    python -m secondlook.features        (with src/ on the path, see README)
"""
from __future__ import annotations

import json
import time

import pandas as pd

from .cleaning import add_application_features, clean_application
from .config import ID_COL, PROCESSED_DIR, TARGET_COL
from .data import load_table, shrink
from .sql_features import attach_history, build_database, history_features

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
META_PATH = PROCESSED_DIR / "features_meta.json"

# Kept in the table for the fairness audit, but never used as model inputs.
PROTECTED_COLS = ["CODE_GENDER"]


def build_feature_table(save: bool = True) -> pd.DataFrame:
    """Full pipeline: load -> clean -> engineer -> SQL history -> join -> save."""
    start = time.time()
    app = load_table("application_train")
    app = clean_application(app)
    income_cap = app.attrs["income_cap"]
    app = add_application_features(app)

    build_database()
    history = history_features()
    features = shrink(attach_history(app, history))

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        features.to_parquet(FEATURES_PATH, index=False)
        META_PATH.write_text(json.dumps({
            "rows": len(features), "columns": features.shape[1],
            "income_cap": income_cap, "id_col": ID_COL, "target_col": TARGET_COL,
            "protected_cols": PROTECTED_COLS,
        }, indent=2))
    print(f"Feature table: {features.shape[0]:,} rows x {features.shape[1]} columns ({time.time() - start:.0f}s)")
    return features


def load_features() -> pd.DataFrame:
    """Load the saved feature table (run build_feature_table first)."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"{FEATURES_PATH} not found. Run notebooks/02_features.ipynb or python -m secondlook.features")
    return pd.read_parquet(FEATURES_PATH)


if __name__ == "__main__":
    build_feature_table()
