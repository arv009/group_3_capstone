"""Build the model-ready feature table: cleaned application data + SQL history summaries.

Run from the project root to rebuild everything from data/raw/:

    python -m secondlook.features        (after `pip install -e .`, see README)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .cleaning import add_application_features, clean_application
from .config import EXCLUDED_INPUTS, ID_COL, PROCESSED_DIR, PROTECTED_COLS, RANDOM_STATE, SPLIT_FRACTIONS, TARGET_COL
from .data import load_table, shrink
from .sql_features import attach_history, history_features

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
SPLITS_PATH = PROCESSED_DIR / "splits.parquet"
META_PATH = PROCESSED_DIR / "features_meta.json"


def build_feature_table(save: bool = True) -> pd.DataFrame:
    """Full pipeline: load -> clean -> engineer -> SQL history -> join -> split -> save."""
    start = time.time()
    app = load_table("application_train")
    app = clean_application(app)
    income_cap = app.attrs["income_cap"]
    app = add_application_features(app)

    history = history_features()
    features = shrink(attach_history(app, history))
    splits = make_splits(features)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        features.to_parquet(FEATURES_PATH, index=False)
        splits.to_parquet(SPLITS_PATH, index=False)
        META_PATH.write_text(json.dumps({
            "rows": len(features), "columns": features.shape[1],
            "income_cap": income_cap, "id_col": ID_COL, "target_col": TARGET_COL,
            "protected_cols": PROTECTED_COLS, "split_fractions": SPLIT_FRACTIONS,
        }, indent=2))
    print(f"Feature table: {features.shape[0]:,} rows x {features.shape[1]} columns ({time.time() - start:.0f}s)")
    return features


def make_splits(features: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Fixed applicant-level split, stratified by default status AND thin-file status,
    so every split has the same share of defaulters and of thin-file applicants."""
    rng = np.random.default_rng(seed)
    names = list(SPLIT_FRACTIONS)
    cum = np.cumsum([SPLIT_FRACTIONS[n] for n in names])
    parts = []
    for _, grp in features.groupby([TARGET_COL, "THIN_FILE"]):
        ids = grp[ID_COL].to_numpy().copy()
        rng.shuffle(ids)
        bounds = (cum * len(ids)).round().astype(int)
        start = 0
        for name, end in zip(names, bounds):
            parts.append(pd.DataFrame({ID_COL: ids[start:end], "split": name}))
            start = end
    return pd.concat(parts, ignore_index=True)


def load_features(with_split: bool = True) -> pd.DataFrame:
    """Load the saved feature table (run build_feature_table first), with its split column."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"{FEATURES_PATH} not found. Run notebooks/02_features.ipynb or python -m secondlook.features")
    df = pd.read_parquet(FEATURES_PATH)
    if with_split:
        df = df.merge(pd.read_parquet(SPLITS_PATH), on=ID_COL, how="left")
    return df


def model_columns(df: pd.DataFrame) -> list[str]:
    """Columns the model may use: everything except IDs, the target, the split, protected attributes
    and EXCLUDED_INPUTS (protected traits or exact proxies, see config.py)."""
    excluded = {ID_COL, TARGET_COL, "split", *PROTECTED_COLS, *EXCLUDED_INPUTS}
    return [c for c in df.columns if c not in excluded]


if __name__ == "__main__":
    build_feature_table()
