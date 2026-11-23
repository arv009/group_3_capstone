"""Payment-history sequence encoder (notebook 07), a SecondLook extra.

sql_features.py summarises each applicant's installments into averages (share paid late, etc.).
Averages throw away ORDER: a borrower who was late years ago but has paid on time since looks
the same as one who has started paying late recently. A small recurrent network (GRU) reads the
last MAX_LEN installments in time order and outputs a "sequence risk score".

Leakage control: the score for a TRAINING applicant comes from a GRU that never saw that
applicant (out-of-fold, like notebook 06). Validation and test applicants get the average score
of the fold models. We stack the score (one number), not the GRU's hidden embedding, because
embeddings from separately trained networks are in different coordinates and cannot be averaged.

This module is the DATA side (DuckDB + NumPy only). The GRU lives in sequence_model.py and is
trained by scripts/train_sequence_scores.py in its own process: on macOS, PyTorch and LightGBM
ship clashing OpenMP runtimes and crash or deadlock when used in the same process.
"""
from __future__ import annotations

import duckdb
import numpy as np
from .data import csv_path

MAX_LEN = 48
STEP_FEATURES = ["days_late", "paid_ratio", "log_amount", "missing_payment", "gap_months", "years_ago", "new_loan"]

SEQUENCE_SQL = """
WITH t AS (
    SELECT SK_ID_CURR, SK_ID_PREV, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT, AMT_INSTALMENT, AMT_PAYMENT,
           ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR
                              ORDER BY DAYS_INSTALMENT DESC, DAYS_ENTRY_PAYMENT DESC NULLS LAST) AS rn,
           DAYS_INSTALMENT - LAG(DAYS_INSTALMENT) OVER w AS gap_days,
           CASE WHEN SK_ID_PREV <> LAG(SK_ID_PREV) OVER w THEN 1 ELSE 0 END AS new_loan
    FROM read_csv_auto('{path}')
    WINDOW w AS (PARTITION BY SK_ID_CURR ORDER BY DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT NULLS LAST)
)
SELECT SK_ID_CURR, rn,
       -- days late (negative = early), capped; a missing payment counts as maximally late
       LEAST(GREATEST(COALESCE(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 60), -30), 60) / 30.0 AS days_late,
       LEAST(GREATEST(COALESCE(AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0), 0), 0), 1.5)     AS paid_ratio,
       LN(1 + AMT_INSTALMENT)                                                           AS log_amount,
       CASE WHEN AMT_PAYMENT IS NULL THEN 1 ELSE 0 END                                  AS missing_payment,
       LEAST(COALESCE(gap_days, 0), 365) / 30.0                                         AS gap_months,
       -DAYS_INSTALMENT / 365.0                                                         AS years_ago,
       new_loan
FROM t
WHERE rn <= {max_len}
"""


def build_sequences(max_len: int = MAX_LEN):
    """Returns (ids, X, lengths): X is float32 [n_applicants, max_len, n_features], oldest first,
    most recent installment at position lengths-1; padding after that is zero."""
    path = csv_path("installments_payments").as_posix().replace("'", "''")
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    res = con.execute(SEQUENCE_SQL.format(path=path, max_len=max_len)).fetchnumpy()
    ids_all = res["SK_ID_CURR"].astype(np.int64)
    rn = res["rn"].astype(np.int64)
    ids, row = np.unique(ids_all, return_inverse=True)
    lengths = np.bincount(row, minlength=len(ids))
    X = np.zeros((len(ids), max_len, len(STEP_FEATURES)), dtype=np.float32)
    pos = lengths[row] - rn  # rn=1 (most recent) goes last
    for j, name in enumerate(STEP_FEATURES):
        X[row, pos, j] = np.asarray(res[name], dtype=np.float32)
    return ids, X, lengths.astype(np.int64)


def standardize(X, lengths, mean=None, std=None):
    """Scale each step feature using statistics over REAL steps only (not padding).
    Pass the training mean/std when scaling validation and test data."""
    mask = np.arange(X.shape[1])[None, :] < lengths[:, None]
    if mean is None:
        real = X[mask]
        mean, std = real.mean(axis=0), real.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xs[~mask] = 0.0
    return Xs.astype(np.float32), mean, std
