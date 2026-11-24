"""Build app/demo_data/analyst.db: the small, read-only SQLite file the AI Analyst queries (Phase 2, Step 12).

It is built only from the committed bundle, so it works on the deployed app (the 2.3 GB database
never goes to the cloud). Three tables:

  portfolio  one row per test applicant (61,503): outcome, loan, decision, PD, group columns, persona
  personas   one row per persona (notebook 09): size, repayment rate, path share, median features
  fairness   one row per group (notebook 08): approval and default rates, equal opportunity, calibration

Run from the project root after notebooks 08 and 09:   python -m secondlook.analyst_db   (with src/ on the path)
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

from .config import ROOT
from .fairness import age_band

BUNDLE = ROOT / "app" / "demo_data"
DB_PATH = BUNDLE / "analyst.db"

SCHEMA = """
portfolio (one row per held-out test applicant, 61,503 rows)
  applicant_id INTEGER       SK_ID_CURR
  defaulted INTEGER          1 = had payment difficulties (default), 0 = repaid
  loan_amount REAL           requested loan amount
  has_bureau_history INTEGER 1 = has a credit bureau record, 0 = thin file
  gender TEXT                'F', 'M' or 'XNA' (never a model input; for fairness questions only)
  age_years REAL
  age_band TEXT              '20s', '30s', '40s', '50s', '60s' (NULL outside 20-69)
  pd REAL                    predicted probability of default (0-1)
  decision TEXT              'approve' (pd < 0.17), 'under_review' (0.17 <= pd < 0.22) or 'decline'
  ext_source_mean REAL       average external credit score (0-1, higher = safer)
  persona TEXT               borrower persona for applicants who are not approved, else NULL
  has_path INTEGER           1 = a smaller loan or longer term would get approved (not-approved applicants only), else NULL

personas (one row per persona of the not-approved applicants)
  persona TEXT, applicants INTEGER, repaid REAL (share 0-1), has_path REAL (share), under_review REAL (share),
  ext_source_mean REAL, credit_to_income REAL, annuity_to_income REAL, credit_term REAL (payments),
  years_employed REAL, inst_late_share REAL, has_bureau_history REAL, bur_n_loans REAL (all medians), sentence TEXT

fairness (one row per group, decisions at the 0.17 cutoff)
  group_type TEXT ('gender', 'age_band', 'credit_file'), group_name TEXT ('women', 'men', '20s', ..., 'thin file',
  'bureau history'), applicants INTEGER, approval_rate REAL, default_rate REAL, repayer_approval REAL (approval
  rate among applicants who repaid: equal opportunity), repayer_decline REAL, default_rate_approved REAL,
  mean_pd REAL, calibration_gap REAL, ratio_to_best REAL, below_four_fifths INTEGER
""".strip()


def build(verbose: bool = True) -> dict:
    port = pd.read_parquet(BUNDLE / "portfolio.parquet")
    cutoff = json.loads((BUNDLE / "model_meta.json").read_text())["cutoff"]
    pers = pd.read_parquet(BUNDLE / "personas.parquet")[["SK_ID_CURR", "persona", "has_path"]]
    portfolio = port.merge(pers, on="SK_ID_CURR", how="left")
    portfolio = pd.DataFrame({
        "applicant_id": portfolio["SK_ID_CURR"], "defaulted": portfolio["TARGET"],
        "loan_amount": portfolio["AMT_CREDIT"], "has_bureau_history": portfolio["HAS_BUREAU_HISTORY"],
        "gender": portfolio["CODE_GENDER"], "age_years": portfolio["AGE_YEARS"].round(1),
        "age_band": age_band(portfolio["AGE_YEARS"]).astype(str).replace("nan", None),
        "pd": portfolio["PD_LGBM"].round(5),
        # the app's three decisions (letters.decision_for): approve, under review (5-point band above the cutoff), decline
        "decision": pd.cut(portfolio["PD_LGBM"], [-1, cutoff, cutoff + 0.05, 2], right=False,
                           labels=["approve", "under_review", "decline"]).astype(str),
        "ext_source_mean": portfolio["EXT_SOURCE_MEAN"].round(4), "persona": portfolio["persona"],
        "has_path": portfolio["has_path"].map({True: 1, False: 0}),
    })
    profile = pd.read_json(BUNDLE / "personas_profile.json").drop(columns=["pca_variance_explained"])
    profile.columns = [c.lower() for c in profile.columns]
    fair = json.loads((BUNDLE / "fairness.json").read_text())
    fairness = pd.concat([pd.DataFrame(rows).assign(group_type=g) for g, rows in fair["groups"].items()])
    fairness = fairness.rename(columns={"group": "group_name"})
    fairness = fairness[["group_type", "group_name"] + [c for c in fairness.columns if c not in ("group_type", "group_name")]]

    DB_PATH.unlink(missing_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        portfolio.to_sql("portfolio", con, index=False)
        profile.to_sql("personas", con, index=False)
        fairness.to_sql("fairness", con, index=False)
        con.execute("CREATE INDEX idx_applicant ON portfolio(applicant_id)")
    report = {"portfolio": len(portfolio), "personas": len(profile), "fairness": len(fairness),
              "size_mb": round(DB_PATH.stat().st_size / 1e6, 2)}
    if verbose:
        print(report)
    return report


if __name__ == "__main__":
    build()
