"""Summarise each applicant's credit history with SQL (DuckDB).

The six history tables hold ~58M rows in total. DuckDB runs SQL directly on the CSV
files, so there is no database to build and each query takes seconds. Every query
returns one row per applicant (SK_ID_CURR), ready to join onto the application table.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from .config import ID_COL
from .data import csv_path

HISTORY_TABLES = [
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "POS_CASH_balance",
    "credit_card_balance",
]

QUERIES: dict[str, str] = {
    # Credit bureau: loans with OTHER lenders. No rows here = thin-file applicant.
    "bureau": """
        SELECT SK_ID_CURR,
               COUNT(*)                                             AS BUREAU_N_LOANS,
               SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS BUREAU_N_ACTIVE,
               SUM(CASE WHEN CREDIT_ACTIVE = 'Bad debt' THEN 1 ELSE 0 END) AS BUREAU_N_BAD_DEBT,
               -MIN(DAYS_CREDIT) / 365.25                           AS BUREAU_YEARS_SINCE_FIRST,
               -MAX(DAYS_CREDIT) / 365.25                           AS BUREAU_YEARS_SINCE_LAST,
               SUM(AMT_CREDIT_SUM_DEBT)                             AS BUREAU_TOTAL_DEBT,
               SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0) AS BUREAU_DEBT_RATIO,
               SUM(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END) AS BUREAU_N_OVERDUE,
               MAX(AMT_CREDIT_MAX_OVERDUE)                          AS BUREAU_MAX_OVERDUE_AMT,
               SUM(CNT_CREDIT_PROLONG)                              AS BUREAU_N_PROLONGED
        FROM bureau
        GROUP BY SK_ID_CURR
    """,
    # Monthly status of bureau loans: share of months that were 1+ months late.
    "bureau_balance": """
        SELECT b.SK_ID_CURR,
               COUNT(*)                                                  AS BB_N_MONTHS,
               AVG(CASE WHEN bb.STATUS IN ('1','2','3','4','5') THEN 1.0 ELSE 0.0 END) AS BB_LATE_SHARE,
               MAX(CASE WHEN bb.STATUS IN ('1','2','3','4','5')
                        THEN CAST(bb.STATUS AS INTEGER) ELSE 0 END)      AS BB_WORST_STATUS
        FROM bureau_balance bb
        JOIN bureau b USING (SK_ID_BUREAU)
        GROUP BY b.SK_ID_CURR
    """,
    # Previous applications to Home Credit, including REFUSED ones (used for reject inference).
    "previous_application": """
        SELECT SK_ID_CURR,
               COUNT(*)                                                       AS PREV_N_APPS,
               SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS PREV_N_APPROVED,
               SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END) AS PREV_N_REFUSED,
               AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1.0 ELSE 0.0 END) AS PREV_REFUSAL_RATE,
               AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                   AS PREV_GRANTED_ASKED_RATIO,
               -MAX(DAYS_DECISION) / 365.25                                   AS PREV_YEARS_SINCE_LAST,
               AVG(CNT_PAYMENT)                                               AS PREV_AVG_TERM
        FROM previous_application
        GROUP BY SK_ID_CURR
    """,
    # Repayment behaviour on previous Home Credit loans: lateness and under-payment.
    "installments_payments": """
        SELECT SK_ID_CURR,
               COUNT(*)                                                         AS INST_N_PAYMENTS,
               AVG(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1.0 ELSE 0.0 END) AS INST_LATE_SHARE,
               AVG(GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0))           AS INST_AVG_DAYS_LATE,
               MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)                        AS INST_MAX_DAYS_LATE,
               AVG(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1.0 ELSE 0.0 END) AS INST_UNDERPAID_SHARE,
               SUM(AMT_PAYMENT) / NULLIF(SUM(AMT_INSTALMENT), 0)                AS INST_PAID_RATIO
        FROM installments_payments
        GROUP BY SK_ID_CURR
    """,
    # Point-of-sale and cash loan monthly snapshots.
    "POS_CASH_balance": """
        SELECT SK_ID_CURR,
               COUNT(*)                                          AS POS_N_MONTHS,
               AVG(SK_DPD)                                       AS POS_AVG_DPD,
               AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END)    AS POS_LATE_SHARE
        FROM POS_CASH_balance
        GROUP BY SK_ID_CURR
    """,
    # Credit card monthly snapshots: utilisation and lateness.
    "credit_card_balance": """
        SELECT SK_ID_CURR,
               COUNT(*)                                                       AS CC_N_MONTHS,
               AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))          AS CC_AVG_UTILISATION,
               MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))          AS CC_MAX_UTILISATION,
               AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END)                 AS CC_LATE_SHARE
        FROM credit_card_balance
        GROUP BY SK_ID_CURR
    """,
}


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with one view per history table, pointing at the CSVs."""
    con = duckdb.connect()
    for name in HISTORY_TABLES:
        path = csv_path(name).as_posix().replace("'", "''")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')")
    return con


def history_features(con: duckdb.DuckDBPyConnection | None = None, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Run every query; returns {table name: one-row-per-applicant DataFrame}."""
    con = con or connect()
    out = {}
    for name, sql in QUERIES.items():
        out[name] = con.execute(sql).df()
        if verbose:
            print(f"  {name:24s} -> {len(out[name]):>9,} applicants x {out[name].shape[1] - 1} features")
    return out


def attach_history(app: pd.DataFrame, history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join all history summaries onto the application table and add segment flags."""
    df = app
    for part in history.values():
        df = df.merge(part, on=ID_COL, how="left")

    # Segment flags. "No rows" is information in itself, so make it explicit.
    df["THIN_FILE"] = df["BUREAU_N_LOANS"].isna().astype("int8")  # no credit bureau history
    df["HAS_PREV_APP"] = df["PREV_N_APPS"].notna().astype("int8")
    df["HAS_REPAYMENT_HISTORY"] = df["INST_N_PAYMENTS"].notna().astype("int8")
    df["PRIOR_REFUSAL"] = (df["PREV_N_REFUSED"].fillna(0) > 0).astype("int8")

    # Counts are genuinely zero (not unknown) when the applicant has no rows.
    count_cols = [c for c in df.columns if "_N_" in c and c.split("_")[0] in {"BUREAU", "BB", "PREV", "INST", "POS", "CC"}]
    df[count_cols] = df[count_cols].fillna(0)
    return df
