"""Load Home Credit tables straight from the Kaggle zip, without extracting it."""
from __future__ import annotations

import zipfile

import pandas as pd

from .config import RAW_ZIP

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


def list_tables(zip_path=RAW_ZIP) -> pd.DataFrame:
    """Return each CSV in the zip with its uncompressed size in MB."""
    with zipfile.ZipFile(zip_path) as z:
        rows = [(i.filename.removesuffix(".csv"), round(i.file_size / 1e6, 1)) for i in z.infolist()]
    return pd.DataFrame(rows, columns=["table", "size_mb"])


def load_table(name: str, zip_path=RAW_ZIP, **read_csv_kwargs) -> pd.DataFrame:
    """Read one table from the zip.

    Pass any pandas.read_csv options, e.g. nrows=1000 or usecols=[...].
    Large tables (installments, bureau_balance, POS_CASH) take a minute and
    a few GB of RAM; use usecols to keep only what you need.
    """
    if name not in TABLES:
        raise ValueError(f"Unknown table {name!r}. Choose one of: {', '.join(TABLES)}")
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found. See README > Data to download it.")
    if name == "HomeCredit_columns_description":
        read_csv_kwargs.setdefault("encoding", "latin-1")
    with zipfile.ZipFile(zip_path) as z, z.open(f"{name}.csv") as f:
        return pd.read_csv(f, **read_csv_kwargs)


def shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to save memory (often 50%+ smaller)."""
    for col in df.select_dtypes("integer").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes("float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df
