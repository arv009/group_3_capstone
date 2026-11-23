"""Explain scores and suggest realistic paths to approval (notebook 04).

Reasons: exact TreeSHAP values computed by LightGBM itself (pred_contrib=True). They are in
log-odds and add up to the model's raw score; isotonic calibration is monotonic, so a feature
that pushes the raw score up also pushes the calibrated probability of default up.

Path to approval: a small, transparent search over ACTIONABLE levers (borrow less, a bigger
down payment, a longer term, paying down debt, lowering card utilisation). Unlike generic
counterfactual tools, each lever changes the raw amounts and then recomputes every ratio
derived from them, so the suggested applicant is internally consistent. Age, gender, past
defaults and anything else outside the applicant's control are never changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- scoring


def score(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Calibrated probability of default."""
    raw = bundle["model"].predict_proba(X[bundle["columns"]])[:, 1]
    return bundle["calibrator"].predict(raw)


def shap_values(bundle: dict, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per-feature contributions in log-odds (positive = raises risk), plus the base value."""
    model = bundle["model"]
    contrib = model.booster_.predict(X[bundle["columns"]], pred_contrib=True, num_iteration=model.best_iteration_)
    return pd.DataFrame(contrib[:, :-1], columns=bundle["columns"], index=X.index), float(contrib[0, -1])


# ----------------------------------------------------------------------------- reasons

# Plain-language description of each feature, phrased neutrally; direction is added from SHAP.
FEATURE_TEXT = {
    "EXT_SOURCE_MEAN": "average external credit score",
    "EXT_SOURCE_MIN": "lowest external credit score",
    "EXT_SOURCE_MAX": "highest external credit score",
    "EXT_SOURCE_1": "external credit score 1",
    "EXT_SOURCE_2": "external credit score 2",
    "EXT_SOURCE_3": "external credit score 3",
    "EXT_SOURCE_N_MISSING": "number of missing external credit scores",
    "CREDIT_ANNUITY_RATIO": "loan size relative to the yearly repayment (loan term)",
    "CREDIT_INCOME_RATIO": "loan size relative to income",
    "ANNUITY_INCOME_RATIO": "yearly repayment relative to income",
    "GOODS_CREDIT_RATIO": "price of the goods relative to the loan",
    "AMT_CREDIT": "loan amount",
    "AMT_ANNUITY": "yearly repayment amount",
    "AMT_GOODS_PRICE": "price of the goods being financed",
    "AMT_INCOME_TOTAL": "income",
    "INCOME_PER_PERSON": "income per household member",
    "YEARS_EMPLOYED": "time in current job",
    "EMPLOYED_AGE_RATIO": "share of adult life in current job",
    "EMPLOYED_ANOMALY": "employment status (pensioner or not employed)",
    "YEARS_SINCE_ID_CHANGE": "time since ID document was last changed",
    "YEARS_SINCE_REGISTRATION": "time since registration was last changed",
    "ORGANIZATION_TYPE": "type of employer",
    "OCCUPATION_TYPE": "occupation",
    "NAME_EDUCATION_TYPE": "education level",
    "NAME_INCOME_TYPE": "income type",
    "NAME_CONTRACT_TYPE": "loan type",
    "NAME_FAMILY_STATUS": "family status",
    "FLAG_OWN_CAR": "car ownership",
    "OWN_CAR_AGE": "how new your car is",
    "REGION_RATING_CLIENT_W_CITY": "region rating",
    "REGION_RATING_CLIENT": "region rating",
    "FLAG_WORK_PHONE": "listing a work phone",
    "FLAG_PHONE": "listing a home phone",
    "FLAG_EMP_PHONE": "listing an employer phone",
    "FLAG_DOCUMENT_3": "providing document 3",
    "REG_CITY_NOT_LIVE_CITY": "living in a different city from the registered address",
    "REG_CITY_NOT_WORK_CITY": "working in a different city from the registered address",
    "CNT_CHILDREN": "number of children",
    "CNT_FAM_MEMBERS": "household size",
    "DEF_30_CNT_SOCIAL_CIRCLE": "defaults in the applicant's social circle",
    "DEF_60_CNT_SOCIAL_CIRCLE": "defaults in the applicant's social circle",
    "NAME_HOUSING_TYPE": "housing situation",
    "FLAG_OWN_REALTY": "home ownership",
    "DAYS_LAST_PHONE_CHANGE": "time since phone number was last changed",
    "N_DOCUMENTS": "number of documents provided",
    "N_BUREAU_ENQUIRIES": "recent credit enquiries",
    "BUREAU_N_LOANS": "number of loans with other lenders",
    "BUREAU_N_ACTIVE": "number of active loans with other lenders",
    "BUREAU_TOTAL_DEBT": "outstanding debt with other lenders",
    "BUREAU_DEBT_RATIO": "share of other-lender credit still owed",
    "BUREAU_YEARS_SINCE_FIRST": "length of credit history",
    "BUREAU_YEARS_SINCE_LAST": "time since the last loan with another lender",
    "BUREAU_N_OVERDUE": "overdue loans with other lenders",
    "BUREAU_MAX_OVERDUE_AMT": "largest overdue amount with other lenders",
    "BB_LATE_SHARE": "share of late months on other-lender loans",
    "PREV_N_APPS": "number of previous applications",
    "PREV_N_REFUSED": "number of previously refused applications",
    "PREV_REFUSAL_RATE": "share of previous applications refused",
    "PREV_GRANTED_ASKED_RATIO": "share of the requested amount granted on previous loans",
    "PREV_AVG_TERM": "term of previous loans",
    "PREV_YEARS_SINCE_LAST": "time since the last previous application",
    "INST_LATE_SHARE": "share of past installments paid late",
    "INST_AVG_DAYS_LATE": "average days late on past installments",
    "INST_MAX_DAYS_LATE": "worst lateness on a past installment",
    "INST_UNDERPAID_SHARE": "share of past installments under-paid",
    "INST_PAID_RATIO": "share of amounts due that were actually paid",
    "INST_N_PAYMENTS": "length of repayment history",
    "POS_N_MONTHS": "length of point-of-sale loan history",
    "POS_LATE_SHARE": "share of months late on point-of-sale loans",
    "CC_AVG_UTILISATION": "average credit card utilisation",
    "CC_MAX_UTILISATION": "highest credit card utilisation",
    "CC_LATE_SHARE": "share of months late on credit cards",
    "THIN_FILE": "having no credit bureau history",
}


def describe(feature: str) -> str:
    return FEATURE_TEXT.get(feature, feature.replace("_", " ").lower())


def top_reasons(contrib_row: pd.Series, x_row: pd.Series, n: int = 4, direction: str = "risk") -> pd.DataFrame:
    """The n features that push this applicant's risk UP the most ("risk") or DOWN the most ("strength")."""
    s = contrib_row.sort_values(ascending=(direction != "risk"))
    s = s[s > 0] if direction == "risk" else s[s < 0]
    s = s.head(n)
    return pd.DataFrame({
        "feature": s.index,
        "value": [x_row[f] for f in s.index],
        "shap": s.values,
        "reason": [describe(f) for f in s.index],
    })


# ----------------------------------------------------------------------------- path to approval

STEPS = np.round(np.arange(0.05, 0.501, 0.05), 2)


@dataclass(frozen=True)
class Lever:
    name: str
    template: str  # how to phrase a change of size k (formatted with k as a percentage)

    def applies(self, x: pd.Series) -> bool:
        if self.name == "pay_down_debt":
            return pd.notna(x.get("BUREAU_TOTAL_DEBT")) and x["BUREAU_TOTAL_DEBT"] > 0
        if self.name == "lower_card_use":
            return pd.notna(x.get("CC_AVG_UTILISATION")) and x["CC_AVG_UTILISATION"] > 0
        return True


LEVERS = [
    Lever("borrow_less", "Choose a {k:.0%} cheaper purchase, so the loan and repayments are {k:.0%} smaller"),
    Lever("bigger_down_payment", "Pay {k:.0%} of the loan upfront as a bigger down payment"),
    Lever("longer_term", "Spread repayments over a term about {k:.0%} longer"),
    Lever("pay_down_debt", "Pay down {k:.0%} of outstanding debt with other lenders"),
    Lever("lower_card_use", "Reduce credit card balances by {k:.0%}"),
]

# "longer_term" is left out by default: the model's response to loan term is jagged (spikes at
# specific product terms, see notebook 04), so moving the term mostly moves the applicant between
# loan products rather than changing their real risk. It can still be requested explicitly.
DEFAULT_LEVERS = ["borrow_less", "bigger_down_payment", "pay_down_debt", "lower_card_use"]


def _join(parts: list[str]) -> str:
    """'Do A' + 'Do B' -> 'Do A and do B'."""
    return " and ".join([parts[0]] + [p[0].lower() + p[1:] for p in parts[1:]])


def _recompute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)
    return df


def apply_lever(df: pd.DataFrame, lever: str, k) -> pd.DataFrame:
    """Apply a lever of size k (scalar or array, one per row) and keep derived features consistent."""
    df = df.copy()
    k = np.asarray(k, dtype=float)
    if lever == "borrow_less":
        for col in ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
            df[col] = df[col] * (1 - k)
    elif lever == "bigger_down_payment":  # same purchase, smaller loan
        df["AMT_CREDIT"] = df["AMT_CREDIT"] * (1 - k)
        df["AMT_ANNUITY"] = df["AMT_ANNUITY"] * (1 - k)
    elif lever == "longer_term":  # same loan, smaller yearly repayment (ignores extra interest)
        df["AMT_ANNUITY"] = df["AMT_ANNUITY"] / (1 + k)
    elif lever == "pay_down_debt":
        df["BUREAU_TOTAL_DEBT"] = df["BUREAU_TOTAL_DEBT"] * (1 - k)
        df["BUREAU_DEBT_RATIO"] = df["BUREAU_DEBT_RATIO"] * (1 - k)
    elif lever == "lower_card_use":
        df["CC_AVG_UTILISATION"] = df["CC_AVG_UTILISATION"] * (1 - k)
        df["CC_MAX_UTILISATION"] = df["CC_MAX_UTILISATION"] * (1 - k)
    else:
        raise ValueError(f"Unknown lever {lever!r}")
    return _recompute_ratios(df)


def path_to_approval(bundle: dict, x: pd.DataFrame, cutoff: float, max_options: int = 3,
                     allow_pairs: bool = True, margin: float = 0.01,
                     levers: list[str] | None = None) -> pd.DataFrame:
    """Smallest changes, on levers the applicant controls, that bring the risk below the cutoff.

    x is a ONE-ROW DataFrame taken from the feature table (so column types are preserved).
    Tries each lever on its own at 5%, 10%, ..., 50%, and pairs of levers at equal size.
    A suggestion must land at least `margin` below the cutoff, so it does not depend on
    scraping just under the line (calibrated scores move in small steps).
    Returns up to max_options rows, single levers first, then the smallest changes.
    """
    target = cutoff - margin
    base = x.reset_index(drop=True)  # full row: levers may touch columns the model does not use
    p0 = float(score(bundle, base)[0])
    row = x.iloc[0]
    allowed = set(levers or DEFAULT_LEVERS)
    levers = [lv for lv in LEVERS if lv.name in allowed and lv.applies(row)]

    # Every (lever combination x step) variant in one batch, scored in a single call.
    combos = [(lv.name,) for lv in levers]
    if allow_pairs:
        combos += [(a.name, b.name) for a, b in combinations(levers, 2)]
    blocks = []
    for names in combos:
        blk = pd.concat([base] * len(STEPS), ignore_index=True)
        for name in names:
            blk = apply_lever(blk, name, STEPS)
        blocks.append(blk)
    p = score(bundle, pd.concat(blocks, ignore_index=True))

    text = {lv.name: lv.template for lv in LEVERS}
    plans = [(names, k) for names in combos for k in STEPS]
    found = []
    for (names, k), pk in zip(plans, p):
        if pk <= target:
            found.append({"levers": names, "size": float(k), "n_levers": len(names), "new_pd": float(pk),
                          "change": _join([text[n].format(k=k) for n in names])})
    if not found:
        return pd.DataFrame(columns=["levers", "size", "n_levers", "new_pd", "change", "current_pd"])

    # Smallest step per lever combination; prefer single levers, then smaller changes.
    opts = (pd.DataFrame(found).sort_values(["n_levers", "size", "new_pd"])
            .drop_duplicates("levers").head(max_options).reset_index(drop=True))
    opts["current_pd"] = p0
    return opts
