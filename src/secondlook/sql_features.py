"""Load the credit-history tables into SQLite and summarise them per applicant with SQL.

Each history table has many rows per applicant (one per past loan, per month or per
installment). A model needs ONE row per applicant, so every query below is a
GROUP BY SK_ID_CURR that turns a history into a handful of numbers.
"""
from __future__ import annotations

import sqlite3
import time
import zipfile
from contextlib import closing

import pandas as pd

from .config import DB_PATH, RAW_ZIP

# Only the columns the queries use: keeps the database small and the load fast.
TABLE_COLUMNS = {
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
               "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
               "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG", "CREDIT_TYPE"],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    "previous_application": ["SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION",
                             "AMT_CREDIT", "AMT_ANNUITY", "CNT_PAYMENT", "DAYS_DECISION", "NAME_CONTRACT_TYPE"],
    "installments_payments": ["SK_ID_PREV", "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                              "AMT_INSTALMENT", "AMT_PAYMENT"],
    "POS_CASH_balance": ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "NAME_CONTRACT_STATUS", "SK_DPD"],
    "credit_card_balance": ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                            "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT", "AMT_PAYMENT_CURRENT",
                            "AMT_INST_MIN_REGULARITY", "SK_DPD"],
}

INDEXES = {
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU"],
    "bureau_balance": ["SK_ID_BUREAU"],
    "previous_application": ["SK_ID_CURR"],
    "installments_payments": ["SK_ID_CURR"],
    "POS_CASH_balance": ["SK_ID_CURR"],
    "credit_card_balance": ["SK_ID_CURR"],
}


def build_database(db_path=DB_PATH, zip_path=RAW_ZIP, force: bool = False, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream each history table from the zip into SQLite, in chunks, and index the join keys.

    Skips tables that are already loaded unless force=True. Returns a row count per table.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if force and db_path.exists():
        db_path.unlink()

    summary = []
    with closing(sqlite3.connect(db_path)) as conn, zipfile.ZipFile(zip_path) as z:
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, cols in TABLE_COLUMNS.items():
            start = time.time()
            if table not in existing:
                with z.open(f"{table}.csv") as f:
                    for chunk in pd.read_csv(f, usecols=cols, chunksize=chunksize):
                        chunk.to_sql(table, conn, if_exists="append", index=False)
                for col in INDEXES[table]:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col})")
                conn.commit()
            rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            summary.append({"table": table, "rows": rows, "columns": len(cols),
                            "load_seconds": round(time.time() - start, 1), "loaded_now": table not in existing})
    return pd.DataFrame(summary).set_index("table")


# ---------------------------------------------------------------------------
# The SQL. One query per history table, each returning one row per SK_ID_CURR.
# Note: in SQLite, a comparison like (x > 0) evaluates to 1 or 0, so AVG(x > 0)
# is "the share of rows where x > 0" and SUM(x > 0) is "how many rows".
# ---------------------------------------------------------------------------
QUERIES = {
    "bureau": """
        -- Loans the applicant has at OTHER lenders (Credit Bureau)
        SELECT
            SK_ID_CURR,
            COUNT(*)                                            AS BUR_N_LOANS,
            SUM(CREDIT_ACTIVE = 'Active')                       AS BUR_N_ACTIVE,
            AVG(CREDIT_ACTIVE = 'Active')                       AS BUR_ACTIVE_SHARE,
            SUM(CREDIT_ACTIVE = 'Bad debt')                     AS BUR_N_BAD_DEBT,
            COUNT(DISTINCT CREDIT_TYPE)                         AS BUR_N_CREDIT_TYPES,
            -MAX(DAYS_CREDIT) / 365.25                          AS BUR_YEARS_SINCE_LAST_LOAN,
            -MIN(DAYS_CREDIT) / 365.25                          AS BUR_YEARS_SINCE_FIRST_LOAN,
            SUM(AMT_CREDIT_SUM)                                 AS BUR_TOTAL_CREDIT,
            SUM(AMT_CREDIT_SUM_DEBT)                            AS BUR_TOTAL_DEBT,
            SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0) AS BUR_DEBT_TO_CREDIT,
            SUM(AMT_CREDIT_SUM_OVERDUE)                         AS BUR_TOTAL_OVERDUE,
            MAX(CREDIT_DAY_OVERDUE)                             AS BUR_MAX_DAYS_OVERDUE,
            SUM(CREDIT_DAY_OVERDUE > 0)                         AS BUR_N_OVERDUE_NOW,
            SUM(CNT_CREDIT_PROLONG)                             AS BUR_N_PROLONGED
        FROM bureau
        GROUP BY SK_ID_CURR
    """,
    "bureau_balance": """
        -- Monthly status of bureau loans. STATUS: 0 = on time, 1-5 = days past due buckets
        -- (1 = 1-30 days ... 5 = 120+ days or written off), C = closed, X = unknown.
        WITH per_loan AS (
            SELECT
                SK_ID_BUREAU,
                COUNT(*)                                        AS n_months,
                SUM(STATUS IN ('1','2','3','4','5'))            AS n_late_months,
                SUM(STATUS IN ('3','4','5'))                    AS n_severe_months
            FROM bureau_balance
            GROUP BY SK_ID_BUREAU
        )
        SELECT
            b.SK_ID_CURR,
            SUM(p.n_months)                                     AS BB_N_MONTHS,
            SUM(p.n_late_months) * 1.0 / SUM(p.n_months)        AS BB_LATE_MONTH_SHARE,
            SUM(p.n_severe_months)                              AS BB_N_SEVERE_LATE_MONTHS
        FROM per_loan p
        JOIN bureau b ON b.SK_ID_BUREAU = p.SK_ID_BUREAU
        GROUP BY b.SK_ID_CURR
    """,
    "previous_application": """
        -- Earlier applications at Home Credit
        SELECT
            SK_ID_CURR,
            COUNT(*)                                            AS PREV_N_APPS,
            SUM(NAME_CONTRACT_STATUS = 'Approved')              AS PREV_N_APPROVED,
            SUM(NAME_CONTRACT_STATUS = 'Refused')               AS PREV_N_REFUSED,
            AVG(NAME_CONTRACT_STATUS = 'Refused')               AS PREV_REFUSED_SHARE,
            AVG(NAME_CONTRACT_STATUS = 'Canceled')              AS PREV_CANCELED_SHARE,
            AVG(AMT_APPLICATION)                                AS PREV_AVG_AMT_APPLIED,
            AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))        AS PREV_AVG_GRANTED_TO_APPLIED,
            AVG(CNT_PAYMENT)                                    AS PREV_AVG_TERM,
            -MAX(DAYS_DECISION) / 365.25                        AS PREV_YEARS_SINCE_LAST_DECISION,
            SUM(NAME_CONTRACT_STATUS = 'Refused' AND DAYS_DECISION >= -365) AS PREV_N_REFUSED_LAST_YEAR
        FROM previous_application
        GROUP BY SK_ID_CURR
    """,
    "installments_payments": """
        -- Every installment due on previous Home Credit loans, and how it was paid.
        -- A missing payment date or amount means the installment was never paid,
        -- so COALESCE counts it as late / underpaid instead of ignoring it.
        SELECT
            SK_ID_CURR,
            COUNT(*)                                            AS INST_N_PAYMENTS,
            AVG(COALESCE(DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT, 1)) AS INST_LATE_SHARE,
            AVG(MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0))   AS INST_AVG_DAYS_LATE,
            MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)           AS INST_MAX_DAYS_LATE,
            AVG(COALESCE(AMT_PAYMENT, 0) < AMT_INSTALMENT)      AS INST_UNDERPAID_SHARE,
            SUM(MAX(AMT_INSTALMENT - COALESCE(AMT_PAYMENT, 0), 0)) AS INST_TOTAL_UNDERPAID,
            AVG(CASE WHEN DAYS_INSTALMENT >= -365
                     THEN COALESCE(DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT, 1) END) AS INST_LATE_SHARE_LAST_YEAR
        FROM installments_payments
        GROUP BY SK_ID_CURR
    """,
    "POS_CASH_balance": """
        -- Monthly snapshots of previous point-of-sale and cash loans
        SELECT
            SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV)                          AS POS_N_LOANS,
            COUNT(*)                                            AS POS_N_MONTHS,
            AVG(SK_DPD > 0)                                     AS POS_LATE_MONTH_SHARE,
            MAX(SK_DPD)                                         AS POS_MAX_DPD,
            COUNT(DISTINCT CASE WHEN NAME_CONTRACT_STATUS = 'Completed' THEN SK_ID_PREV END) AS POS_N_COMPLETED
        FROM POS_CASH_balance
        GROUP BY SK_ID_CURR
    """,
    "credit_card_balance": """
        -- Monthly snapshots of previous Home Credit credit cards
        SELECT
            SK_ID_CURR,
            COUNT(DISTINCT SK_ID_PREV)                          AS CC_N_CARDS,
            COUNT(*)                                            AS CC_N_MONTHS,
            AVG(AMT_BALANCE)                                    AS CC_AVG_BALANCE,
            AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS CC_AVG_UTILISATION,
            AVG(AMT_DRAWINGS_CURRENT)                           AS CC_AVG_DRAWINGS,
            AVG(AMT_PAYMENT_CURRENT < AMT_INST_MIN_REGULARITY)  AS CC_BELOW_MIN_PAYMENT_SHARE,
            AVG(SK_DPD > 0)                                     AS CC_LATE_MONTH_SHARE,
            MAX(SK_DPD)                                         AS CC_MAX_DPD
        FROM credit_card_balance
        GROUP BY SK_ID_CURR
    """,
}

# Count columns mean "how many": an applicant with no rows has 0, not "unknown".
COUNT_PREFIXES = ("BUR_N_", "BB_N_", "PREV_N_", "INST_N_", "POS_N_", "CC_N_")


def run_query(name: str, db_path=DB_PATH) -> pd.DataFrame:
    """Run one of the QUERIES and return it as a DataFrame indexed by SK_ID_CURR."""
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql_query(QUERIES[name], conn, index_col="SK_ID_CURR")


def history_features(db_path=DB_PATH, verbose: bool = True) -> pd.DataFrame:
    """Run every query and join the results into one table (one row per applicant)."""
    parts = []
    for name in QUERIES:
        start = time.time()
        part = run_query(name, db_path)
        if verbose:
            print(f"{name:24s} {len(part):>9,} applicants  {part.shape[1]:>2} features  {time.time() - start:5.1f}s")
        parts.append(part)
    return pd.concat(parts, axis=1)


def attach_history(app: pd.DataFrame, history: pd.DataFrame, id_col: str = "SK_ID_CURR") -> pd.DataFrame:
    """LEFT JOIN the history features onto the application table and fill counts with 0."""
    out = app.merge(history, left_on=id_col, right_index=True, how="left")
    count_cols = [c for c in history.columns if c.startswith(COUNT_PREFIXES)]
    out[count_cols] = out[count_cols].fillna(0)
    out["HAS_BUREAU_HISTORY"] = out["BUR_N_LOANS"].gt(0).astype("int8")
    out["HAS_PREV_APPLICATION"] = out["PREV_N_APPS"].gt(0).astype("int8")
    return out
