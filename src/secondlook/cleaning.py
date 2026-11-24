"""Clean the main application table, following the decisions made in notebooks/01_eda.ipynb."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_EMPLOYED_PLACEHOLDER = 365243

# Building-detail columns come in three near-identical versions (_AVG, _MODE, _MEDI).
# EDA showed they are missing together and almost perfectly correlated, so we keep only _AVG.
_BUILDING_SUFFIXES = ("_MODE", "_MEDI")
_KEEP_MODE_COLS = {"FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE",
                   "EMERGENCYSTATE_MODE", "TOTALAREA_MODE"}  # categorical / no _AVG twin


def clean_application(df: pd.DataFrame, income_cap: float | None = None) -> pd.DataFrame:
    """Return a cleaned copy of application_train (or application_test).

    Steps (each one traced back to an EDA finding):
      1. DAYS_EMPLOYED = 365243 is a placeholder for "not employed" (18% of rows, mostly
         pensioners) -> set to NaN and keep the information in a flag column.
      2. CODE_GENDER "XNA" (4 rows) -> NaN.
      3. Cap AMT_INCOME_TOTAL at the 99.9th percentile to neutralise the 117M data-entry error.
      4. Convert negative day counts to positive years (easier to read and explain).
      5. Missing OCCUPATION_TYPE -> "Unknown" (31% missing; "no occupation recorded" is information).
      6. Drop the duplicate _MODE / _MEDI building columns.

    Pass income_cap from the training data when cleaning new applicants, so the
    same cap is used everywhere.
    """
    df = df.copy()

    # 1. Employment placeholder
    anomaly = df["DAYS_EMPLOYED"].eq(DAYS_EMPLOYED_PLACEHOLDER)
    df["DAYS_EMPLOYED_ANOM"] = anomaly.astype("int8")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].where(~anomaly)

    # 2. Unknown gender code
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    # 3. Income cap
    if income_cap is None:
        income_cap = df["AMT_INCOME_TOTAL"].quantile(0.999)
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=income_cap)

    # 4. Days -> years (all DAYS_* are "days before the application")
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].abs() / 365.25
    df["YEARS_SINCE_REGISTRATION"] = df["DAYS_REGISTRATION"].abs() / 365.25
    df["YEARS_SINCE_ID_PUBLISH"] = df["DAYS_ID_PUBLISH"].abs() / 365.25
    df["YEARS_SINCE_PHONE_CHANGE"] = df["DAYS_LAST_PHONE_CHANGE"].abs() / 365.25
    df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
                          "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"])

    # 5. Occupation
    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

    # 6. Duplicate building columns
    dup = [c for c in df.columns if c.endswith(_BUILDING_SUFFIXES) and c not in _KEEP_MODE_COLS]
    df = df.drop(columns=dup)

    df.attrs["income_cap"] = float(income_cap)
    return df


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio and summary features to a cleaned application table.

    The ratios came out of EDA section 10.3: lenders judge a loan relative to the
    applicant's income and to the price of what is being bought, not in absolute terms.
    Several of these (loan amount, term, loan-to-goods) are the "levers" SecondLook
    later uses to suggest a realistic path to approval.
    """
    df = df.copy()
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]

    # Loan structure
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TO_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]          # number of payments, roughly
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    # External scores: the strongest signals in EDA
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_MAX"] = ext.max(axis=1)
    df["EXT_SOURCE_STD"] = ext.std(axis=1)
    df["EXT_SOURCE_N_MISSING"] = ext.isna().sum(axis=1).astype("int8")

    # Missingness itself is a signal (thin files have fewer scores)
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_3", "AMT_GOODS_PRICE"]:
        df[f"{col}_MISSING"] = df[col].isna().astype("int8")

    # Employment relative to age
    df["EMPLOYED_TO_AGE"] = df["YEARS_EMPLOYED"] / df["AGE_YEARS"]

    # Documents provided and contact details given
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["N_DOCUMENTS"] = df[doc_cols].sum(axis=1).astype("int8")
    contact_cols = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
    df["N_CONTACTS"] = df[contact_cols].sum(axis=1).astype("int8")

    # Credit bureau enquiries in the last year (more enquiries = shopping for credit)
    enq = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
           "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]
    df["BUREAU_ENQUIRIES_YEAR"] = df[enq].sum(axis=1, min_count=1)

    return df.replace([np.inf, -np.inf], np.nan)
