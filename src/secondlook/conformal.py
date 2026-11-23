"""Uncertainty-aware decisions: approve, decline or refer to an underwriter (notebook 08).

Two layers, both fit on the "calib" split, which the LightGBM model never saw. Use the model's
RAW scores here: the isotonic calibrator was fit on "calib", so calibrated scores are not
"unseen" there.

1. Conformal prediction sets (class-conditional, "Mondrian"). For each applicant, the set of
   outcomes we cannot rule out at confidence 1 - alpha: {repay}, {default} or {repay, default}.
   Guarantee: on new applicants, the set contains the true outcome with probability >= 1 - alpha,
   separately for repayers and for defaulters.
2. Decision-level referral. Bin applicants by score (bin edges fixed on another split), put a
   simultaneous Clopper-Pearson interval on each bin's default rate, and compare it with the
   break-even default rate of the loan economics: approve when the whole interval is below,
   decline when it is above, refer when it straddles (the data cannot tell whether lending is profitable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta


def _conformal_quantile(scores, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score: the finite-sample conformal threshold."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= len(s) else np.inf


def marginal_threshold(y, p, alpha: float) -> float:
    """Standard split conformal: one threshold for the nonconformity 1 - P(true label).
    Coverage holds on average over everyone, which on imbalanced data mostly means repayers."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    return _conformal_quantile(np.where(y == 1, 1 - p, p), alpha)


def class_conditional_thresholds(y, p, alpha: float) -> tuple[float, float]:
    """Mondrian conformal: separate thresholds so coverage holds for repayers AND for defaulters.
    Nonconformity of 'repay' is p (the default score); of 'default' it is 1 - p."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    return _conformal_quantile(p[y == 0], alpha), _conformal_quantile(1 - p[y == 1], alpha)


def prediction_sets(p, q_repay: float, q_default: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays: does each applicant's set include 'repay', and 'default'?"""
    p = np.asarray(p, dtype=float)
    return p <= q_repay, (1 - p) <= q_default


def set_decision(has_repay, has_default) -> np.ndarray:
    """{repay} -> approve, {default} -> decline, both or neither -> refer."""
    has_repay, has_default = np.asarray(has_repay, bool), np.asarray(has_default, bool)
    return np.select([has_repay & ~has_default, has_default & ~has_repay], ["approve", "decline"], "refer")


def bin_edges(p_reference, n_bins: int = 20) -> np.ndarray:
    """Equal-frequency score bins from a split OTHER than the one used for the intervals."""
    inner = np.unique(np.quantile(np.asarray(p_reference, dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def binned_intervals(y, p, edges, confidence: float = 0.95) -> pd.DataFrame:
    """Default rate per score bin with Clopper-Pearson intervals that hold for ALL bins at once
    (Bonferroni: each bin gets 1 - (1 - confidence) / n_bins)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    n_bins = len(edges) - 1
    a = (1 - confidence) / n_bins
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        yk = y[b == k]
        n, d = len(yk), int(yk.sum())
        low = beta.ppf(a / 2, d, n - d + 1) if d > 0 else 0.0
        high = beta.ppf(1 - a / 2, d + 1, n - d) if d < n else 1.0
        rows.append({"bin": k, "lower_edge": edges[k], "upper_edge": edges[k + 1], "n": n,
                     "default_rate": d / n if n else np.nan, "low": low, "high": high})
    return pd.DataFrame(rows).set_index("bin")


def assign_zones(intervals: pd.DataFrame, breakeven_pd: float) -> pd.DataFrame:
    """Approve bins whose interval lies below break-even, decline bins above it, refer the rest.
    Zones are made monotone in the score: approve only up to the first bin that is not a clear
    approve, decline only from the last bin that is not a clear decline."""
    out = intervals.copy()
    clear_approve = (out["high"] < breakeven_pd).to_numpy()
    clear_decline = (out["low"] > breakeven_pd).to_numpy()
    n = len(out)
    n_approve = int(np.argmin(clear_approve)) if not clear_approve.all() else n
    n_decline = int(np.argmin(clear_decline[::-1])) if not clear_decline.all() else n
    zone = np.full(n, "refer", dtype=object)
    zone[:n_approve] = "approve"
    zone[n - n_decline:] = "decline"
    out["zone"] = zone
    return out


def decide(p, zones: pd.DataFrame) -> np.ndarray:
    """Approve / decline / refer for new applicants, from their raw score and the zone table."""
    edges = np.concatenate([zones["lower_edge"].to_numpy(), [zones["upper_edge"].iloc[-1]]])
    b = np.clip(np.searchsorted(edges, np.asarray(p, dtype=float), side="right") - 1, 0, len(zones) - 1)
    return zones["zone"].to_numpy()[b]
