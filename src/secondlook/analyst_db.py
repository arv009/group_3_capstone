"""Build app/demo_data/analyst.db: a small read-only SQLite file the AI Analyst queries.

Made only from the app bundle (never the 2.3 GB working database), so it can ship to the cloud:

    portfolio      one row per test applicant (61,503): outcome, PD, decision, groups, persona, path
    personas       one row per persona (5): size, repayment rate, path share, typical features
    fairness       one row per audit group (9): approval, default and calibration metrics
    fairness_gaps  one row per gap (12): bootstrap 95% intervals

Run:  python -m secondlook.analyst_db      (with src/ on the path)
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from .config import ROOT

BUNDLE = ROOT / "app" / "demo_data"
DB_PATH = BUNDLE / "analyst.db"
AGE_BINS, AGE_LABELS = [20, 30, 40, 50, 60, 70], ["20-29", "30-39", "40-49", "50-59", "60-69"]


def portfolio_table() -> pd.DataFrame:
    meta = json.loads((BUNDLE / "model_meta.json").read_text())
    p = pd.read_parquet(BUNDLE / "portfolio.parquet")
    pers = pd.read_parquet(BUNDLE / "personas.parquet")[["SK_ID_CURR", "persona", "has_path"]]
    demo = set(pd.read_parquet(BUNDLE / "applicants.parquet")["SK_ID_CURR"])
    lo, hi = meta["borderline_band"]
    out = pd.DataFrame({
        "applicant_id": p["SK_ID_CURR"].astype(int),
        "defaulted": p["TARGET"].astype(int),
        "loan_amount": p["AMT_CREDIT"].astype(float).round(0),
        "pd": p["PD_LGBM"].astype(float).round(5),
        "decision": np.select([p["PD_LGBM"] < lo, p["PD_LGBM"] < hi], ["approve", "under_review"], "decline"),
        "has_bureau_record": p["HAS_BUREAU_HISTORY"].astype(int),
        "gender": p["CODE_GENDER"].map({"F": "female", "M": "male"}),
        "age": p["AGE_YEARS"].astype(float).round(1),
        "age_band": pd.cut(p["AGE_YEARS"], AGE_BINS, right=False, labels=AGE_LABELS).astype(str).replace("nan", None),
        "ext_score": p["EXT_SOURCE_MEAN"].astype(float).round(4),
    })
    out = out.merge(pers.rename(columns={"SK_ID_CURR": "applicant_id"}), on="applicant_id", how="left")
    out["has_path"] = out["has_path"].map({True: 1, False: 0})            # NULL for approved applicants
    out["in_demo"] = out["applicant_id"].isin(demo).astype(int)
    return out


def personas_table() -> pd.DataFrame:
    from .personas import PE_COLUMNS, describe, profile
    pers = pd.read_parquet(BUNDLE / "personas.parquet")
    prof = profile(pers)
    out = prof.rename(columns=PE_COLUMNS).reset_index()
    out["description"] = [describe(n, r) for n, r in prof.iterrows()]
    return out


def fairness_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    f = json.loads((BUNDLE / "fairness.json").read_text())
    rows = []
    for grouping, recs in f["tables"].items():
        for r in recs:
            rows.append({"grouping": grouping.lower().replace(" ", "_"), "grp": r["Group"], "applicants": r["Applicants"],
                         "default_rate": r["Default rate"], "approval_rate": r["Approval rate"],
                         "approval_rate_repayers": r["Approval rate of repayers"],
                         "decline_rate_repayers": r["Decline rate of repayers"],
                         "default_rate_approved": r["Default rate of approved"], "mean_pd": r["Mean predicted PD"],
                         "calibration_gap": r["Calibration gap"], "four_fifths_ratio": r["Four-fifths ratio"]})
    gaps = pd.DataFrame(f["gaps"]).rename(columns={"Grouping": "grouping", "Comparison": "comparison", "Metric": "metric",
                                                   "Gap": "gap", "95% CI low": "ci_low", "95% CI high": "ci_high",
                                                   "Real (CI excludes no gap)": "real"})
    gaps["grouping"] = gaps["grouping"].str.lower().str.replace(" ", "_")
    gaps["real"] = gaps["real"].astype(int)
    return pd.DataFrame(rows), gaps


def build(path=DB_PATH) -> dict:
    """(Re)build the database and return the row count of each table."""
    path.unlink(missing_ok=True)
    fair, gaps = fairness_tables()
    tables = {"portfolio": portfolio_table(), "personas": personas_table(), "fairness": fair, "fairness_gaps": gaps}
    with sqlite3.connect(path) as con:
        for name, df in tables.items():
            df.to_sql(name, con, index=False)
        con.execute("CREATE INDEX idx_decision ON portfolio(decision)")
        con.execute("CREATE INDEX idx_persona ON portfolio(persona)")
        con.execute("CREATE UNIQUE INDEX idx_id ON portfolio(applicant_id)")
    with sqlite3.connect(path) as con:
        con.execute("VACUUM")
    return {k: len(v) for k, v in tables.items()}


if __name__ == "__main__":
    counts = build()
    print(f"Built {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB): {counts}")
