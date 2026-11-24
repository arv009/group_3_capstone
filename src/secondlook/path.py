"""Path to approval: the smallest realistic change to the loan that brings an applicant under the cutoff.

SecondLook only changes what the applicant and lender can actually renegotiate, the loan itself:
  * the loan amount (borrow less, in 5% steps, up to 40% less)
  * the number of payments (a longer term, capped at what approved applicants typically get)

Everything else about the applicant (credit history, profile, sensitive traits) stays exactly as it is.
After each change, the features built from the loan terms are recomputed the same way
cleaning.add_application_features builds them, and the trained model re-scores the applicant.

Important limits (also stated in notebook 06 and the app):
  * The model learned associations from historical data, not cause and effect. A lower re-scored PD
    means "applicants with loans like this repaid more often", not a guarantee.
  * The regular payment is estimated as loan amount / number of payments (the same simple definition
    the model's CREDIT_TERM feature uses); a real offer would add interest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import modeling as M

CUTS = tuple(round(0.05 * i, 2) for i in range(9))          # 0%, 5%, ..., 40% less
TERM_MULTS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)                  # current term up to 50% longer
# 95th percentile of CREDIT_TERM among approved training applicants (computed and checked in notebook 06)
TERM_CAP = 34.7
# A suggested change must bring PD at least this far under the cutoff, so an offer never hinges on
# the third decimal place (a PD of 0.1699 would show as "17.0%" next to a 17% cutoff)
MARGIN = 0.005
# Features that depend on the loan terms; nothing else may change
LOAN_FEATURES = ["AMT_CREDIT", "AMT_ANNUITY", "CREDIT_TERM", "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TO_GOODS"]


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    """PD from either the scikit-learn LGBMClassifier (notebooks) or a plain Booster (app)."""
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


def apply_terms(rows: pd.DataFrame, cut: float, term_mult: float) -> pd.DataFrame:
    """Copy of rows with the loan cut by `cut` and the term multiplied by `term_mult`; loan features recomputed."""
    r = rows.copy()
    credit = r["AMT_CREDIT"] * (1 - cut)
    term = r["CREDIT_TERM"] * term_mult
    r["AMT_CREDIT"] = credit
    r["AMT_ANNUITY"] = credit / term
    r["CREDIT_TERM"] = term
    r["CREDIT_TO_INCOME"] = credit / r["AMT_INCOME_TOTAL"]
    r["ANNUITY_TO_INCOME"] = r["AMT_ANNUITY"] / r["AMT_INCOME_TOTAL"]
    r["CREDIT_TO_GOODS"] = credit / r["AMT_GOODS_PRICE"]
    return r.replace([np.inf, -np.inf], np.nan)


def score_grid(rows: pd.DataFrame, model, meta: dict, cuts=CUTS, term_mults=TERM_MULTS,
               term_cap: float = TERM_CAP) -> pd.DataFrame:
    """Re-score every applicant in `rows` under every (cut, term) option. One row per applicant x option.

    Options that are not realistic are left out: a term above term_cap (unless it is the current term),
    and any term change on a revolving loan (a credit line has no fixed number of payments).
    """
    feats, cats, categories = meta["features"], meta["categorical"], meta["categories"]
    pa = meta["profit_assumptions"]
    out = []
    for m in term_mults:
        for c in cuts:
            r = apply_terms(rows, c, m)
            ok = (m == 1.0) | ((r["CREDIT_TERM"] <= term_cap) & (rows["NAME_CONTRACT_TYPE"] != "Revolving loans"))
            if not ok.any():
                continue
            r = r[ok]
            p = _predict(model, M.prepare_X(r, feats, cats, categories))
            out.append(pd.DataFrame({
                "SK_ID_CURR": r["SK_ID_CURR"].values, "cut": c, "term_mult": m,
                "loan_amount": r["AMT_CREDIT"].values, "regular_payment": r["AMT_ANNUITY"].values,
                "n_payments": r["CREDIT_TERM"].values, "pd": p,
                # expected profit per loan under the Phase 1 assumptions (+margin if repaid, -LGD if not)
                "expected_profit": r["AMT_CREDIT"].values * ((1 - p) * pa["margin"] - p * pa["lgd"]),
            }))
    g = pd.concat(out, ignore_index=True)
    # Size of the change: share of the loan given up + share of extra payments (a 10% cut counts like
    # a 10% longer term). The recommended option is the smallest change; profit only breaks ties.
    g["change"] = (g["cut"] + (g["term_mult"] - 1)).round(2)
    return g


def _rank(options: pd.DataFrame) -> pd.DataFrame:
    """Smallest change first; among equal changes, the higher expected profit."""
    return options.sort_values(["change", "expected_profit"], ascending=[True, False])


def summarise(grid: pd.DataFrame, cutoff: float, margin: float = MARGIN) -> pd.DataFrame:
    """One row per applicant: current PD and the best option of each kind (None/NaN when there is none).

    borrow_less   smallest cut that is approved at the current term
    longer_term   shortest term increase that is approved at the full amount
    recommended   the approved option with the smallest change (ties: higher expected profit)
    """
    g = grid.copy()
    g["approved"] = g["pd"] < cutoff - margin
    base = g[(g["cut"] == 0) & (g["term_mult"] == 1.0)].set_index("SK_ID_CURR")["pd"].rename("pd_now")
    ok = g[g["approved"] & ~((g["cut"] == 0) & (g["term_mult"] == 1.0))]
    less = ok[ok["term_mult"] == 1.0].sort_values("cut").groupby("SK_ID_CURR").first()["cut"].rename("borrow_less_cut")
    longer = ok[ok["cut"] == 0].sort_values("term_mult").groupby("SK_ID_CURR").first()["term_mult"].rename("longer_term_mult")
    rec = _rank(ok).groupby("SK_ID_CURR").first()
    rec = rec[["cut", "term_mult", "pd"]].add_prefix("rec_")
    s = pd.concat([base, less, longer, rec], axis=1)
    s["has_path"] = s["rec_pd"].notna()
    s.index.name = "SK_ID_CURR"
    return s.reset_index()


def path_to_approval(row: pd.DataFrame, model, meta: dict, cutoff: float, term_cap: float = TERM_CAP,
                     margin: float = MARGIN) -> dict:
    """Path to approval for ONE applicant (a one-row DataFrame with all model features).

    Returns {"status": "approved" | "path" | "no_path", "current": {...}, "options": DataFrame of approved
    options (best first), "borrow_less": {...} | None, "longer_term": {...} | None, "recommended": {...} | None}.
    """
    grid = score_grid(row, model, meta, term_cap=term_cap)
    current = grid[(grid["cut"] == 0) & (grid["term_mult"] == 1.0)].iloc[0].to_dict()
    if current["pd"] < cutoff:
        return {"status": "approved", "current": current, "options": grid.iloc[0:0],
                "borrow_less": None, "longer_term": None, "recommended": None}
    ok = _rank(grid[grid["pd"] < cutoff - margin]).reset_index(drop=True)
    pick = lambda t: t.iloc[0].to_dict() if len(t) else None
    return {
        "status": "path" if len(ok) else "no_path",
        "current": current,
        "options": ok,
        "borrow_less": pick(ok[ok["term_mult"] == 1.0].sort_values("cut")),
        "longer_term": pick(ok[ok["cut"] == 0].sort_values("term_mult")),
        "recommended": pick(ok),
    }
