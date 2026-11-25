"""Borrower personas: group the applicants the model does not approve into a few recognisable types.

K-Means on 8 interpretable, non-sensitive features, inside a scikit-learn Pipeline:
median imputation -> StandardScaler -> KMeans. Scaling matters: without it, the feature with the
largest numbers (loan-to-income in the tens, the number of past loans) would decide the clusters alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_DIR, RANDOM_STATE

PERSONA_FEATURES = ["EXT_SOURCE_MEAN", "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TERM",
                    "YEARS_EMPLOYED", "INST_LATE_SHARE", "HAS_BUREAU_HISTORY", "BUR_N_LOANS"]
FEATURE_LABELS = {
    "EXT_SOURCE_MEAN": "External credit score", "CREDIT_TO_INCOME": "Loan ÷ yearly income",
    "ANNUITY_TO_INCOME": "Payment ÷ income", "CREDIT_TERM": "Number of payments",
    "YEARS_EMPLOYED": "Years in current job", "INST_LATE_SHARE": "Share of past payments late",
    "HAS_BUREAU_HISTORY": "Has a bureau record", "BUR_N_LOANS": "Loans at other lenders",
}
PATHS_PATH = PROCESSED_DIR / "paths_not_approved.parquet"
# profile() columns -> SQL-friendly names for the AI Analyst's database
PE_COLUMNS = {
    "Applicants": "applicants", "Share": "share_of_not_approved", "EXT_SOURCE_MEAN": "median_ext_score",
    "CREDIT_TO_INCOME": "median_loan_to_income", "ANNUITY_TO_INCOME": "median_payment_to_income",
    "CREDIT_TERM": "median_n_payments", "YEARS_EMPLOYED": "median_years_employed",
    "INST_LATE_SHARE": "median_late_payment_share", "HAS_BUREAU_HISTORY": "share_with_bureau_record",
    "BUR_N_LOANS": "median_loans_elsewhere", "Repayment rate": "repayment_rate", "Mean PD": "mean_pd",
    "Path to approval": "share_with_path",
}


def make_pipeline(k: int) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("kmeans", KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10))])


def scan_k(X: pd.DataFrame, ks=range(3, 9)) -> pd.DataFrame:
    """Inertia (for the elbow plot) and silhouette score for each k."""
    rows = []
    for k in ks:
        pipe = make_pipeline(k).fit(X)
        Z = pipe[:-1].transform(X)
        labels = pipe[-1].labels_
        rows.append({"k": k, "inertia": pipe[-1].inertia_, "silhouette": silhouette_score(Z, labels),
                     "smallest cluster": int(np.bincount(labels).min())})
    return pd.DataFrame(rows).set_index("k")


def profile(df: pd.DataFrame, label_col: str = "persona") -> pd.DataFrame:
    """Per persona: size, share, median of each feature, actual repayment rate, share with a path to approval."""
    g = df.groupby(label_col, observed=True)
    t = g[PERSONA_FEATURES].median()
    t["HAS_BUREAU_HISTORY"] = g["HAS_BUREAU_HISTORY"].mean()           # a share, more useful than a 0/1 median
    t.insert(0, "Applicants", g.size())
    t.insert(1, "Share", t["Applicants"] / len(df))
    t["Repayment rate"] = 1 - g["TARGET"].mean()
    t["Mean PD"] = g["PD"].mean()
    t["Path to approval"] = g["has_path"].mean()
    return t


def name_personas(prof: pd.DataFrame) -> dict:
    """Name each cluster from its profile only (never age or gender), one distinguishing trait each:
    the cluster with the fewest bureau records is the thin file, the highest late share the missed payments,
    the highest loan-to-income the big loan, the longest job tenure the long-tenure group; the rest is the
    largest, most ordinary group. Assumes 5 clusters."""
    left, names = list(prof.index), {}
    for col, pick, name in [("HAS_BUREAU_HISTORY", "idxmin", "Thin file"),
                            ("INST_LATE_SHARE", "idxmax", "Missed payments before"),
                            ("CREDIT_TO_INCOME", "idxmax", "Big loan for the income"),
                            ("YEARS_EMPLOYED", "idxmax", "Long job, many past loans")]:
        c = getattr(prof.loc[left, col], pick)()
        names[c] = name
        left.remove(c)
    assert len(left) == 1, "name_personas expects 5 clusters"
    names[left[0]] = "Newer job, clean record"
    return names


def describe(name: str, r: pd.Series) -> str:
    """One plain-English sentence per persona, built from its profile numbers."""
    tail = (f"{r['Repayment rate']:.0%} actually repaid, and {r['Path to approval']:.0%} have a path to approval "
            f"by borrowing less or taking more payments.")
    lead = {
        "Thin file": "No record at any credit bureau, so the lender can't read them; otherwise ordinary loans.",
        "Missed payments before": f"Paid {r['INST_LATE_SHARE']:.0%} of past instalments late (median), "
                                  "but have the best external scores of the five groups.",
        "Big loan for the income": f"Asking for {r['CREDIT_TO_INCOME']:.1f}x their yearly income, with payments taking "
                                   f"{r['ANNUITY_TO_INCOME']:.0%} of income over about {r['CREDIT_TERM']:.0f} payments.",
        "Long job, many past loans": f"About {r['YEARS_EMPLOYED']:.0f} years in their job and {r['BUR_N_LOANS']:.0f} loans "
                                     "at other lenders: experienced, stable borrowers.",
        "Newer job, clean record": f"The largest group: about {r['YEARS_EMPLOYED']:.0f} years in their job, few late payments "
                                   "and ordinary loans; not approved on scores rather than any single red flag.",
    }[name]
    return f"{lead} {tail}"


def pca_map(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """2D PCA coordinates of the scaled features, and the share of variance each component explains."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(pipe[:-1].transform(X))
    return pca.transform(pipe[:-1].transform(X)), pca.explained_variance_ratio_


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Strength of association between two categorical variables, 0 (none) to 1 (one decides the other)."""
    from scipy.stats import chi2_contingency
    t = pd.crosstab(a, b)
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))
