"""Load the Home Credit tables from data/raw/.

Works with either layout:
  * the Kaggle zip kept as is (data/raw/home-credit-default-risk.zip), or
  * the extracted CSVs (data/raw/*.csv), which is what scripts/download_data.py produces.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .config import INTERIM_DIR, RAW_DIR, RAW_ZIP

TABLES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "POS_CASH_balance",
    "credit_card_balance",
    "installments_payments",
    "HomeCredit_columns_description",
]


def _check(name: str) -> None:
    if name not in TABLES:
        raise ValueError(f"Unknown table {name!r}. Choose one of: {', '.join(TABLES)}")


def list_tables() -> pd.DataFrame:
    """Return each available table with its uncompressed size in MB."""
    if RAW_ZIP.exists():
        with zipfile.ZipFile(RAW_ZIP) as z:
            rows = [(i.filename.removesuffix(".csv"), i.file_size / 1e6) for i in z.infolist()]
    else:
        rows = [(p.stem, p.stat().st_size / 1e6) for p in sorted(RAW_DIR.glob("*.csv"))]
    return pd.DataFrame(rows, columns=["table", "size_mb"]).round(1)


def load_table(name: str, **read_csv_kwargs) -> pd.DataFrame:
    """Read one table. Pass any pandas.read_csv options, e.g. nrows=1000 or usecols=[...].

    The large tables (installments, bureau_balance, POS_CASH) take a minute and a few
    GB of RAM; use usecols to keep only what you need.
    """
    _check(name)
    if name == "HomeCredit_columns_description":
        read_csv_kwargs.setdefault("encoding", "latin-1")
    csv = RAW_DIR / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv, **read_csv_kwargs)
    if RAW_ZIP.exists():
        with zipfile.ZipFile(RAW_ZIP) as z, z.open(f"{name}.csv") as f:
            return pd.read_csv(f, **read_csv_kwargs)
    raise FileNotFoundError(f"No {name}.csv or Kaggle zip in {RAW_DIR}. See README > Data.")


def csv_path(name: str) -> Path:
    """Path to a table as a plain CSV on disk (DuckDB needs a real file).

    If only the zip is present, the table is extracted once into data/interim/.
    """
    _check(name)
    csv = RAW_DIR / f"{name}.csv"
    if csv.exists():
        return csv
    out = INTERIM_DIR / f"{name}.csv"
    if not out.exists():
        if not RAW_ZIP.exists():
            raise FileNotFoundError(f"No {name}.csv or Kaggle zip in {RAW_DIR}. See README > Data.")
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RAW_ZIP) as z:
            z.extract(f"{name}.csv", INTERIM_DIR)
    return out


def shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to save memory (often 50%+ smaller)."""
    for col in df.select_dtypes("integer").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes("float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df
