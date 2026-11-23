"""Reject inference (notebook 06), a SecondLook extra.

A lender only learns outcomes for loans it GRANTED. A model trained on those alone has never
seen the kind of applicant the lender used to turn down, and thin-file applicants are turned
down more often, so the bias lands exactly where SecondLook wants to help.

Reject inference adds the declined applicants back into training with INFERRED outcomes:

  * Fuzzy augmentation: each declined applicant is added twice, once as a default (weight p)
    and once as a repayment (weight 1 - p), where p is the approved-only model's estimate.
  * Parcelling: declined applicants are grouped into score bands and given random outcomes
    at the band's observed default rate times an inflation factor (declined applicants are
    assumed to be riskier than approved ones with the same score).

Because real declined applicants never have outcomes, the methods are evaluated with a
SIMULATION on data where every outcome is known (see simulate_policy and notebook 06).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_policy(policy_scores, reject_share: float) -> tuple[np.ndarray, float]:
    """An existing lender approves everyone except the reject_share with the highest risk score
    under its current model. Returns (approve mask, score cutoff); apply the same cutoff to
    other splits so the policy is identical everywhere."""
    policy_scores = np.asarray(policy_scores)
    cutoff = float(np.quantile(policy_scores, 1 - reject_share))
    return policy_scores <= cutoff, cutoff


def out_of_fold_scores(X: pd.DataFrame, y, n_estimators: int, params: dict, n_folds: int = 5,
                       seed: int = 42) -> np.ndarray:
    """Risk scores for the rows a model was trained on, without leakage: each fold is scored by a
    model fitted on the other folds. Needed when simulating a lender's policy on training data,
    because in-sample scores would 'know' who defaulted."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    out = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_idx, score_idx in folds.split(X, y):
        m = lgb.LGBMClassifier(**(params | {"n_estimators": n_estimators}))
        m.fit(X.iloc[fit_idx], y[fit_idx])
        out[score_idx] = m.predict_proba(X.iloc[score_idx])[:, 1]
    return out


def fuzzy_augment(X_acc: pd.DataFrame, y_acc, X_rej: pd.DataFrame, p_rej, reject_weight: float = 1.0):
    """Approved rows (weight 1) + each rejected row twice: as default with weight p, as repayment
    with weight 1 - p, both scaled by reject_weight. Returns (X, y, w)."""
    p_rej = np.clip(np.asarray(p_rej, dtype=float), 0, 1)
    X = pd.concat([X_acc, X_rej, X_rej], ignore_index=True)
    y = np.concatenate([np.asarray(y_acc), np.ones(len(X_rej), int), np.zeros(len(X_rej), int)])
    w = np.concatenate([np.ones(len(X_acc)), reject_weight * p_rej, reject_weight * (1 - p_rej)])
    return X, y, w


def parcel(X_acc: pd.DataFrame, y_acc, p_acc, X_rej: pd.DataFrame, p_rej, n_bands: int = 10,
           inflation: float = 1.5, seed: int = 42):
    """Give each rejected applicant a random outcome at (approved default rate in its score band)
    x inflation. Bands are score deciles of the APPROVED applicants; rejects above the top band
    use the top band. Returns (X, y, w) with all weights 1."""
    rng = np.random.default_rng(seed)
    y_acc, p_acc, p_rej = np.asarray(y_acc), np.asarray(p_acc), np.asarray(p_rej)
    edges = np.unique(np.quantile(p_acc, np.linspace(0, 1, n_bands + 1)))
    band_acc = np.clip(np.searchsorted(edges, p_acc, side="right") - 1, 0, len(edges) - 2)
    band_rej = np.clip(np.searchsorted(edges, p_rej, side="right") - 1, 0, len(edges) - 2)
    band_rate = pd.Series(y_acc).groupby(band_acc).mean().reindex(range(len(edges) - 1)).ffill().bfill().to_numpy()
    p_bad = np.clip(band_rate[band_rej] * inflation, 0, 1)
    y_rej = (rng.random(len(X_rej)) < p_bad).astype(int)
    X = pd.concat([X_acc, X_rej], ignore_index=True)
    y = np.concatenate([y_acc, y_rej])
    return X, y, np.ones(len(X))


def approval_at_bad_rate(y, p, thin, target: float) -> dict:
    """Using the test labels, approve the lowest-risk applicants until the approved book's
    default rate would exceed target. Returns overall and thin-file approval rates.
    (An oracle cutoff: it isolates ranking quality from cutoff-selection noise.)"""
    y, thin = np.asarray(y), np.asarray(thin, dtype=bool)
    order = np.argsort(p, kind="stable")
    bad_rate = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.nonzero(bad_rate <= target)[0]
    n = ok[-1] + 1 if len(ok) else 0
    approved = np.zeros(len(y), bool)
    approved[order[:n]] = True
    return {"approval_rate": approved.mean(), "thin_approval_rate": approved[thin].mean()}
