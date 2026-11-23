"""Fairness audit and mitigation across gender and age bands (notebook 08).

Gender and age are never model inputs (features.model_columns); they are used here only to audit
decisions. Convention: label y = 1 means default; a decision of 1 (or a probability) means approve.

Metrics per group (Fairlearn MetricFrame):
  * approval rate                      (demographic parity compares these)
  * good-borrower approval rate        (equal opportunity: creditworthy people approved equally)
  * default rate among approved        (does the cutoff mean the same risk in every group?)
  * mean predicted vs observed default (calibration within group)

Mitigation helpers: per-group cutoffs that equalise good-borrower approval (post-processing,
the idea behind Fairlearn's ThresholdOptimizer), blended with the single cutoff to trace the
fairness / profit trade-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame

AGE_BANDS = [18, 30, 40, 50, 60, 70]


def age_band(age_years: pd.Series) -> pd.Series:
    return pd.cut(age_years, bins=AGE_BANDS, right=False)


def _good_approval(y, a):
    y, a = np.asarray(y), np.asarray(a, dtype=float)
    return a[y == 0].mean()


def _bad_among_approved(y, a):
    y, a = np.asarray(y), np.asarray(a, dtype=float)
    return (a * y).sum() / a.sum() if a.sum() else np.nan


def group_report(y, approve, groups, p=None) -> pd.DataFrame:
    """Decision metrics per group (Fairlearn MetricFrame), plus an 'overall' row.
    `approve` may be 0/1 or an approval probability; `p` (predicted default) adds a calibration check."""
    metrics = {
        "applicants": lambda y, a: len(y),
        "default rate": lambda y, a: np.mean(y),
        "approval rate": lambda y, a: np.mean(a),
        "good-borrower approval": _good_approval,
        "default rate among approved": _bad_among_approved,
    }
    groups = pd.Series(np.asarray(groups), name="group")
    frames = [MetricFrame(metrics=metrics, y_true=np.asarray(y), y_pred=np.asarray(approve, dtype=float),
                          sensitive_features=groups)]
    if p is not None:
        frames.append(MetricFrame(metrics={"mean predicted default": lambda y, q: np.mean(q)},
                                  y_true=np.asarray(y), y_pred=np.asarray(p, dtype=float), sensitive_features=groups))
    by_group = pd.concat([f.by_group for f in frames], axis=1)
    overall = pd.concat([f.overall for f in frames]).rename("overall").to_frame().T
    return pd.concat([by_group, overall]).astype({"applicants": int})


def equal_opportunity_gap(y, approve, groups) -> float:
    """Largest difference in good-borrower approval rate between any two groups."""
    df = pd.DataFrame({"y": np.asarray(y), "a": np.asarray(approve, dtype=float), "g": np.asarray(groups)})
    rates = df[df["y"] == 0].groupby("g", observed=True)["a"].mean()
    return float(rates.max() - rates.min())


def parity_thresholds(y, p, groups, cutoff: float) -> dict:
    """Per-group cutoffs so that every group's good-borrower approval rate equals the overall rate at
    `cutoff`. Fit on a labelled split (e.g. valid), apply to new applicants."""
    y, p, groups = np.asarray(y), np.asarray(p, dtype=float), np.asarray(groups)
    good = y == 0
    target = (p[good] <= cutoff).mean()
    return {g: float(np.quantile(p[good & (groups == g)], target)) for g in pd.unique(groups)}


def blend_thresholds(thresholds: dict, cutoff: float, strength: float) -> dict:
    """strength 0 = one cutoff for everyone, 1 = full parity thresholds, in between = partial."""
    return {g: (1 - strength) * cutoff + strength * t for g, t in thresholds.items()}


def apply_thresholds(p, groups, thresholds: dict, default_cutoff: float) -> np.ndarray:
    groups = np.asarray(groups)
    cut = np.array([thresholds.get(g, default_cutoff) for g in groups], dtype=float)
    return np.asarray(p, dtype=float) <= cut


def portfolio_profit(y, approve, amount, margin: float, lgd: float) -> float:
    """Expected profit of a set of decisions (approve may be a probability): margin on repaid loans,
    minus lgd on defaulted ones, as a share of the credit amount."""
    y, a, amount = np.asarray(y), np.asarray(approve, dtype=float), np.asarray(amount, dtype=float)
    return float((a * amount * np.where(y == 1, -lgd, margin)).sum())
