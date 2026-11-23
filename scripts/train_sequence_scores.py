"""Train the payment-history GRU out-of-fold and save one sequence risk score per applicant.

Runs in its own process because PyTorch and LightGBM clash on macOS (see sequence_model.py).
Notebook 07 calls this script, then uses the scores in LightGBM.

Usage:  python scripts/train_sequence_scores.py
Writes: data/processed/sequence_scores.parquet (SK_ID_CURR, split, SEQ_SCORE)
        data/processed/sequence_training.json  (validation AUC per fold and epoch, runtime)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # import secondlook from src/

import numpy as np
import pandas as pd
import torch

from secondlook.config import ID_COL, PROCESSED_DIR, TARGET_COL
from secondlook.features import load_features
from secondlook.sequence import build_sequences, standardize
from secondlook.sequence_model import out_of_fold_scores

torch.set_num_threads(4)
start = time.time()

ids, X_raw, lengths = build_sequences()
feat = load_features()[[ID_COL, TARGET_COL, "split"]]
seq = feat.merge(pd.DataFrame({ID_COL: ids, "row": np.arange(len(ids))}), on=ID_COL, how="inner")
print(f"{len(seq):,} labelled applicants with installment history", flush=True)


def part(split):
    s = seq[seq["split"] == split]
    return s["row"].to_numpy(), s[TARGET_COL].to_numpy(), s[ID_COL].to_numpy()


(r_tr, y_tr, id_tr), (r_va, y_va, id_va) = part("train"), part("valid")
(r_te, _, id_te), (r_ca, _, id_ca) = part("test"), part("calib")

X_tr, mean, std = standardize(X_raw[r_tr], lengths[r_tr])  # scaling learned on training applicants only
X_va = standardize(X_raw[r_va], lengths[r_va], mean, std)[0]
X_te = standardize(X_raw[r_te], lengths[r_te], mean, std)[0]
X_ca = standardize(X_raw[r_ca], lengths[r_ca], mean, std)[0]
del X_raw

oof, outside, histories = out_of_fold_scores(
    X_tr, lengths[r_tr], y_tr, X_va, lengths[r_va], y_va,
    others={"test": (X_te, lengths[r_te]), "calib": (X_ca, lengths[r_ca])}, epochs=6, patience=1)

scores = pd.concat([
    pd.DataFrame({ID_COL: id_tr, "split": "train", "SEQ_SCORE": oof}),
    pd.DataFrame({ID_COL: id_va, "split": "valid", "SEQ_SCORE": outside["valid"]}),
    pd.DataFrame({ID_COL: id_te, "split": "test", "SEQ_SCORE": outside["test"]}),
    pd.DataFrame({ID_COL: id_ca, "split": "calib", "SEQ_SCORE": outside["calib"]}),
], ignore_index=True)
scores.to_parquet(PROCESSED_DIR / "sequence_scores.parquet", index=False)
minutes = (time.time() - start) / 60
(PROCESSED_DIR / "sequence_training.json").write_text(json.dumps(
    {"valid_auc_per_fold_and_epoch": histories, "minutes": round(minutes, 1)}, indent=2))
print(f"Saved {len(scores):,} sequence scores in {minutes:.1f} min; best valid AUC per fold: "
      + ", ".join(f"{max(h):.3f}" for h in histories), flush=True)
