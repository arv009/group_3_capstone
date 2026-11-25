"""Fairness audit: are good borrowers treated equally across gender, age and thin files?

Everything here works on test-set predictions (one row per applicant with TARGET, a PD column and the
group columns). The questions, per group:

  * approval rate                 how often the group is approved
  * default rate                  how often the group actually defaults (the base rate)
  * approval rate of repayers     among applicants who repaid, how many were approved ("equal opportunity")
  * decline rate of repayers      1 - the above: good borrowers turned away
  * default rate of approved      do approved loans in this group default more often? (risk is shared fairly)
  * mean PD vs actual rate        calibration: does the model over- or under-state the group's risk?
  * four-fifths ratio             group approval rate / best group's approval rate; under 0.8 is a red flag

Gaps are bootstrapped (resampling applicants with replacement) to give 95% intervals, so a gap can be
called real or noise. The fix experiment retrains LightGBM without the sensitive features.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from . import modeling as M
from .config import MODELS_DIR, PROCESSED_DIR, TARGET_COL

# Sensitive personal traits the model may use only indirectly. Gender is already excluded from training.
SENSITIVE_FEATURES = ["AGE_YEARS", "EMPLOYED_TO_AGE", "NAME_FAMILY_STATUS", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]
AGE_BINS, AGE_LABELS = [20, 30, 40, 50, 60, 70], ["20-29", "30-39", "40-49", "50-59", "60-69"]
FOUR_FIFTHS = 0.8
MIN_GROUP = 100           # smaller groups (e.g. 2 applicants with unknown gender) are left out

NO_SENSITIVE_MODEL = MODELS_DIR / "lgbm_no_sensitive.joblib"
NO_SENSITIVE_PREDS = PROCESSED_DIR / "test_predictions_no_sensitive.parquet"


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three audit groupings as readable labels: Gender, Age band, Credit file."""
    return df.assign(
        Gender=df["CODE_GENDER"].map({"F": "Female", "M": "Male"}),
        **{"Age band": pd.cut(df["AGE_YEARS"], AGE_BINS, right=False, labels=AGE_LABELS)},
        **{"Credit file": np.where(df["HAS_BUREAU_HISTORY"].eq(1), "Bureau history", "Thin file")},
    )


def group_table(df: pd.DataFrame, group: str, pd_col: str = "PD_LGBM", cutoff: float = 0.17) -> pd.DataFrame:
    """One row per group with every audit metric (see the module docstring)."""
    d = df[[group, TARGET_COL, pd_col]].dropna(subset=[group]).copy()
    d["approved"] = d[pd_col] < cutoff
    g = d.groupby(group, observed=True)
    rep = d[d[TARGET_COL] == 0].groupby(group, observed=True)["approved"]
    t = pd.DataFrame({
        "Applicants": g.size(),
        "Default rate": g[TARGET_COL].mean(),
        "Approval rate": g["approved"].mean(),
        "Approval rate of repayers": rep.mean(),
        "Decline rate of repayers": 1 - rep.mean(),
        "Default rate of approved": d[d["approved"]].groupby(group, observed=True)[TARGET_COL].mean(),
        "Mean predicted PD": g[pd_col].mean(),
    })
    t = t[t["Applicants"] >= MIN_GROUP]
    t["Calibration gap"] = t["Mean predicted PD"] - t["Default rate"]        # > 0: model overstates the risk
    t["Four-fifths ratio"] = t["Approval rate"] / t["Approval rate"].max()
    t["Below 0.8"] = t["Four-fifths ratio"] < FOUR_FIFTHS
    return t


def gap_pair(table: pd.DataFrame) -> tuple[str, str]:
    """The groups compared for a grouping's headline gap: highest vs lowest approval rate."""
    order = table["Approval rate"].sort_values(ascending=False, kind="stable").index   # two different groups, even on a tie
    return order[0], order[-1]


GAP_METRICS = ["Approval rate", "Approval rate of repayers", "Default rate of approved", "Four-fifths ratio"]


def bootstrap_gaps(df: pd.DataFrame, group: str, pd_col: str = "PD_LGBM", cutoff: float = 0.17,
                   n_boot: int = 1000, seed: int = 42, pair: tuple[str, str] | None = None) -> pd.DataFrame:
    """Gap between two groups (best vs worst approval rate unless `pair` is given) for each GAP_METRIC,
    with a 95% bootstrap interval from resampling applicants with replacement.

    For the four-fifths ratio the "gap" is the worst group's approval rate / best group's."""
    hi, lo = pair or gap_pair(group_table(df, group, pd_col, cutoff))
    d = df[df[group].isin([hi, lo])]
    is_hi = d[group].eq(hi).to_numpy()
    y = d[TARGET_COL].to_numpy().astype(bool)
    ok = (d[pd_col] < cutoff).to_numpy()

    def gaps(idx):
        # Count-based means for each resample (rows = resamples), so 1,000 resamples run in seconds
        h, yy, aa = is_hi[idx], y[idx], ok[idx]
        def rate(num, den):
            return num.sum(axis=-1) / den.sum(axis=-1)
        out = {}
        a_hi, a_lo = rate(aa & h, h), rate(aa & ~h, ~h)
        out["Approval rate"] = a_hi - a_lo
        out["Approval rate of repayers"] = rate(aa & h & ~yy, h & ~yy) - rate(aa & ~h & ~yy, ~h & ~yy)
        out["Default rate of approved"] = rate(yy & aa & h, aa & h) - rate(yy & aa & ~h, aa & ~h)
        out["Four-fifths ratio"] = a_lo / a_hi
        return out

    point = gaps(np.arange(len(d))[None, :])
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in GAP_METRICS}
    for start in range(0, n_boot, 100):                       # 100 resamples at a time keeps memory small
        idx = rng.integers(0, len(d), size=(min(100, n_boot - start), len(d)))
        for k, v in gaps(idx).items():
            draws[k].append(v)
    rows = []
    for k in GAP_METRICS:
        v = np.concatenate(draws[k])
        lo_ci, hi_ci = np.percentile(v, [2.5, 97.5])
        zero = 1.0 if k == "Four-fifths ratio" else 0.0
        rows.append({"Grouping": group, "Comparison": f"{hi} vs {lo}", "Metric": k, "Gap": float(point[k][0]),
                     "95% CI low": lo_ci, "95% CI high": hi_ci,
                     "Real (CI excludes no gap)": not (lo_ci <= zero <= hi_ci)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fix experiment: retrain without the sensitive features
# ---------------------------------------------------------------------------
def retrain_without(features: pd.DataFrame, drop: list[str] = SENSITIVE_FEATURES, save: bool = True) -> dict:
    """Retrain LightGBM exactly like notebook 03 (same split, settings, 5-fold CV for the tree count,
    trees = mean best round x 1.1), but without `drop`. Returns test predictions and CV results."""
    start = time.time()
    feats, cats = M.model_columns(features)
    feats = [f for f in feats if f not in drop]
    cats = [c for c in cats if c not in drop]
    train, test = M.split(features)
    categories = {c: sorted(train[c].dropna().unique().tolist()) for c in cats}
    X_train, X_test = M.prepare_X(train, feats, cats, categories), M.prepare_X(test, feats, cats, categories)
    cv = M.cross_validate_lgbm(X_train, train[TARGET_COL])
    n_trees = int(np.mean(cv["best_rounds"]) * 1.1)
    model = M.fit_lgbm(X_train, train[TARGET_COL], n_trees)
    preds = pd.DataFrame({"SK_ID_CURR": test["SK_ID_CURR"].to_numpy(),
                          "PD_NO_SENSITIVE": model.predict_proba(X_test)[:, 1]})
    out = {"preds": preds, "cv_auc_mean": cv["mean_auc"], "cv_auc_std": cv["std_auc"], "n_trees": n_trees,
           "n_features": len(feats), "dropped": list(drop), "seconds": round(time.time() - start)}
    if save:
        joblib.dump({k: v for k, v in out.items() if k != "preds"} | {"model": model}, NO_SENSITIVE_MODEL)
        preds.to_parquet(NO_SENSITIVE_PREDS, index=False)
    return out


def load_no_sensitive() -> tuple[pd.DataFrame, dict]:
    """Saved predictions and training summary of the retrained model (run retrain_without first)."""
    info = joblib.load(NO_SENSITIVE_MODEL)
    return pd.read_parquet(NO_SENSITIVE_PREDS), {k: v for k, v in info.items() if k != "model"}


def model_summary(y, pd_hat, amount, cutoff: float) -> dict:
    """AUC, profit, approval and default rates for one model at one cutoff."""
    s = M.scores(y, pd_hat)
    curve = M.profit_curve(y, pd_hat, amount, cutoffs=[cutoff]).iloc[0]
    return {"ROC AUC": s["roc_auc"], "Brier score": s["brier"], "Cutoff": cutoff,
            "Approval rate": curve["approval_rate"], "Default rate of approved": curve["default_rate_approved"],
            "Good borrowers declined": int(curve["good_declined"]), "Profit (billions)": curve["profit"] / 1e9}


def best_cutoff(y, pd_hat, amount) -> float:
    """Profit-maximising cutoff, chosen the same way as for the shipped model."""
    curve = M.profit_curve(y, pd_hat, amount)
    return float(curve.loc[curve["profit"].idxmax(), "cutoff"])
