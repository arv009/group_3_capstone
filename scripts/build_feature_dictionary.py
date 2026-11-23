"""Write docs/feature_dictionary.csv: every column in the feature table, where it comes
from, what it means, and whether the model may use it.

Usage:  python scripts/build_feature_dictionary.py   (after building the feature table)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # import secondlook from src/

import pandas as pd

from secondlook.config import ID_COL, PROTECTED_COLS, ROOT, TARGET_COL
from secondlook.data import load_table
from secondlook.features import load_features, model_columns

ENGINEERED = {
    "EMPLOYED_ANOMALY": "1 if DAYS_EMPLOYED held the 365243 placeholder (almost all pensioners)",
    "AGE_YEARS": "Applicant age in years (protected: fairness audit only)",
    "YEARS_EMPLOYED": "Years in current employment (missing if placeholder)",
    "YEARS_SINCE_ID_CHANGE": "Years since the applicant last changed their ID document",
    "YEARS_SINCE_REGISTRATION": "Years since the applicant last changed their registration",
    "CREDIT_INCOME_RATIO": "Loan amount / annual income",
    "ANNUITY_INCOME_RATIO": "Loan annuity / annual income (debt-service burden)",
    "CREDIT_ANNUITY_RATIO": "Loan amount / annuity (proxy for loan term)",
    "GOODS_CREDIT_RATIO": "Price of goods / loan amount",
    "INCOME_PER_PERSON": "Annual income / family members",
    "EMPLOYED_AGE_RATIO": "Years employed / age",
    "EXT_SOURCE_MEAN": "Mean of the three external scores",
    "EXT_SOURCE_MIN": "Minimum of the three external scores",
    "EXT_SOURCE_MAX": "Maximum of the three external scores",
    "EXT_SOURCE_N_MISSING": "Number of external scores missing (0-3)",
    "N_DOCUMENTS": "Number of documents provided (sum of FLAG_DOCUMENT_*)",
    "N_BUREAU_ENQUIRIES": "Total credit bureau enquiries before application",
    "THIN_FILE": "1 if the applicant has no credit bureau history (SecondLook's target segment)",
    "HAS_PREV_APP": "1 if the applicant applied to Home Credit before",
    "HAS_REPAYMENT_HISTORY": "1 if the applicant has installment repayment records",
    "PRIOR_REFUSAL": "1 if any previous Home Credit application was refused",
}

SOURCES = {
    "BUREAU_": "bureau (SQL)", "BB_": "bureau_balance (SQL)", "PREV_": "previous_application (SQL)",
    "INST_": "installments_payments (SQL)", "POS_": "POS_CASH_balance (SQL)", "CC_": "credit_card_balance (SQL)",
}

df = load_features(with_split=False)
raw_desc = load_table("HomeCredit_columns_description")
raw_desc = raw_desc[raw_desc["Table"] == "application_{train|test}.csv"].set_index("Row")["Description"]
allowed = set(model_columns(df))

rows = []
for col in df.columns:
    source = next((s for p, s in SOURCES.items() if col.startswith(p)), None)
    if col in ENGINEERED:
        source, desc = "engineered (cleaning.py / sql_features.py)", ENGINEERED[col]
    elif source:
        desc = "History summary; see the query in src/secondlook/sql_features.py"
    else:
        source, desc = "application_train", raw_desc.get(col, "")
    role = ("id" if col == ID_COL else "target" if col == TARGET_COL
            else "protected (audit only)" if col in PROTECTED_COLS else "model input" if col in allowed else "excluded")
    rows.append({"column": col, "source": source, "role": role, "dtype": str(df[col].dtype),
                 "missing_share": round(float(df[col].isna().mean()), 4), "description": desc})

out = ROOT / "docs" / "feature_dictionary.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Wrote {len(rows)} columns to {out.relative_to(ROOT)}")
