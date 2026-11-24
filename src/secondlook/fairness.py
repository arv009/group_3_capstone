"""Fairness audit helpers (Phase 2, Step 10; notebook 08). Plain pandas and numpy.

Groups: gender, age band (20s to 60s) and thin file (no credit bureau record) vs bureau history.
"Approved" means PD under the cutoff (0.17), the model's own decision before any officer review.

Metrics per group (all shares are 0-1):
  approval_rate          share approved
  default_rate           share who actually defaulted (TARGET = 1)
  repayer_approval       approval rate among applicants who repaid: EQUAL OPPORTUNITY. A fair
                         model approves good borrowers at the same rate in every group.
  repayer_decline        1 - repayer_approval: good borrowers turned away
  default_rate_approved  share of the approved who defaulted (is the group's approval "earned"?)
  mean_pd, calibration_gap  mean predicted PD and predicted minus actual default rate (calibration)
  ratio_to_best          approval rate / the best group's approval rate. The "four-fifths rule of thumb"
                         from US employment law flags a ratio under 0.8 as possible adverse impact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGE_BANDS = [20, 30, 40, 50, 60, 70]
AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]
FOUR_FIFTHS = 0.8


def age_band(age_years: pd.Series) -> pd.Series:
    """20s ... 60s; the handful of applicants under 20 or 70 and over fall outside the bands (NaN)."""
    return pd.cut(age_years, AGE_BANDS, right=False, labels=AGE_LABELS)


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three audit groupings to a frame with CODE_GENDER, AGE_YEARS and HAS_BUREAU_HISTORY."""
    out = df.copy()
    out["gender"] = out["CODE_GENDER"].map({"F": "women", "M": "men"})        # 'XNA' (4 applicants) left out
    out["age_band"] = age_band(out["AGE_YEARS"])
    out["credit_file"] = np.where(out["HAS_BUREAU_HISTORY"] == 1, "bureau history", "thin file")
    return out


def group_metrics(df: pd.DataFrame, group: str, pd_col: str = "PD_LGBM", cutoff: float = 0.17) -> pd.DataFrame:
    """One row per group value with every metric in the module docstring."""
    d = df.dropna(subset=[group]).assign(approved=lambda x: x[pd_col] < cutoff, repaid=lambda x: x["TARGET"] == 0)
    g = d.groupby(group, observed=True)
    out = pd.DataFrame({
        "applicants": g.size(),
        "approval_rate": g["approved"].mean(),
        "default_rate": g["TARGET"].mean(),
        "repayer_approval": d[d["repaid"]].groupby(group, observed=True)["approved"].mean(),
        "default_rate_approved": d[d["approved"]].groupby(group, observed=True)["TARGET"].mean(),
        "mean_pd": g[pd_col].mean(),
    })
    out["repayer_decline"] = 1 - out["repayer_approval"]
    out["calibration_gap"] = out["mean_pd"] - out["default_rate"]
    out["ratio_to_best"] = out["approval_rate"] / out["approval_rate"].max()
    out["below_four_fifths"] = out["ratio_to_best"] < FOUR_FIFTHS
    return out


def bootstrap_gap(df: pd.DataFrame, group: str, a: str, b: str, pd_col: str = "PD_LGBM", cutoff: float = 0.17,
                  metric: str = "repayer_approval", n_boot: int = 1000, seed: int = 42) -> dict:
    """Gap = metric(group a) - metric(group b), with a 95% bootstrap interval (resampling applicants).

    metric: "repayer_approval" (equal opportunity) or "approval_rate"."""
    d = df[df[group].isin([a, b])]
    if metric == "repayer_approval":
        d = d[d["TARGET"] == 0]
    approved = (d[pd_col] < cutoff).to_numpy(dtype=float)
    in_a = (d[group] == a).to_numpy()
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        x, ga = approved[idx], in_a[idx]
        gaps[i] = x[ga].mean() - x[~ga].mean()
    point = approved[in_a].mean() - approved[~in_a].mean()
    low, high = np.percentile(gaps, [2.5, 97.5])
    return {"group": group, "a": a, "b": b, "metric": metric, "gap": point, "low": low, "high": high,
            "real": bool(low > 0 or high < 0)}       # the interval excludes zero: not just noise
