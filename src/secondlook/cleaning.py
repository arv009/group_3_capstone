"""Clean the main application table and add applicant-level engineered features."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_EMPLOYED_PLACEHOLDER = 365243  # ~1,000 years; used for pensioners / unemployed


def clean_application(app: pd.DataFrame, income_cap: float | None = None) -> pd.DataFrame:
    """Fix known data problems. Returns a copy; the income cap used is stored in df.attrs.

    Pass the income_cap learned on the training data when cleaning new applicants,
    so a single extreme value can't shift the cap.
    """
    df = app.copy()

    # 1. DAYS_EMPLOYED placeholder -> missing, with a flag (the flag itself is predictive).
    df["EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_PLACEHOLDER).astype("int8")
    df.loc[df["EMPLOYED_ANOMALY"] == 1, "DAYS_EMPLOYED"] = np.nan

    # 2. Gender "XNA" (4 rows) -> missing. Gender is kept only for the fairness audit.
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    # 3. Cap extreme incomes at the 99.9th percentile (a few are in the hundreds of millions).
    if income_cap is None:
        income_cap = float(df["AMT_INCOME_TOTAL"].quantile(0.999))
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=income_cap)

    # 4. Negative "days before application" -> positive years, easier to read and explain.
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25
    df["YEARS_SINCE_ID_CHANGE"] = -df["DAYS_ID_PUBLISH"] / 365.25
    df["YEARS_SINCE_REGISTRATION"] = -df["DAYS_REGISTRATION"] / 365.25
    df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION"])

    df.attrs["income_cap"] = income_cap
    return df


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios and summaries a credit analyst would compute by hand."""
    df = df.copy()
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    credit = df["AMT_CREDIT"].replace(0, np.nan)

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income  # debt-service burden
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)  # proxy for loan term
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / credit
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["EMPLOYED_AGE_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE_YEARS"]

    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
    df["EXT_SOURCE_MIN"] = df[ext].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[ext].max(axis=1)
    df["EXT_SOURCE_N_MISSING"] = df[ext].isna().sum(axis=1).astype("int8")

    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["N_DOCUMENTS"] = df[doc_cols].sum(axis=1).astype("int8")

    enquiry_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    df["N_BUREAU_ENQUIRIES"] = df[enquiry_cols].sum(axis=1, min_count=1)
    return df
