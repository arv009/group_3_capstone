"""Unit tests for cleaning, splitting and the protected-attribute guard.

Run from the project root:  pytest
These tests use small hand-made tables, so they run without the Kaggle data.
"""
import numpy as np
import pandas as pd
import pytest

from secondlook.cleaning import DAYS_EMPLOYED_PLACEHOLDER, add_application_features, clean_application
from secondlook.config import EXCLUDED_INPUTS, ID_COL, PROTECTED_COLS, SPLIT_FRACTIONS, TARGET_COL
from secondlook.features import make_splits, model_columns


def _app(n=4):
    return pd.DataFrame({
        ID_COL: range(n),
        TARGET_COL: [0, 1, 0, 0][:n],
        "CODE_GENDER": ["F", "M", "XNA", "F"][:n],
        "DAYS_BIRTH": [-10000, -15000, -20000, -25000][:n],
        "DAYS_EMPLOYED": [-1000, DAYS_EMPLOYED_PLACEHOLDER, -3000, -400][:n],
        "DAYS_ID_PUBLISH": [-100] * n,
        "DAYS_REGISTRATION": [-200.0] * n,
        "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, 0.0, 1e9][:n],
        "AMT_CREDIT": [500_000.0] * n,
        "AMT_ANNUITY": [25_000.0, 0.0, 20_000.0, 30_000.0][:n],
        "AMT_GOODS_PRICE": [450_000.0] * n,
        "CNT_FAM_MEMBERS": [2.0, 1.0, 0.0, 3.0][:n],
        "EXT_SOURCE_1": [0.5, np.nan, np.nan, 0.2][:n],
        "EXT_SOURCE_2": [0.6, 0.4, np.nan, 0.3][:n],
        "EXT_SOURCE_3": [0.7, 0.1, np.nan, np.nan][:n],
        "FLAG_DOCUMENT_3": [1, 0, 1, 1][:n],
        "AMT_REQ_CREDIT_BUREAU_YEAR": [1.0, np.nan, 2.0, 0.0][:n],
    })


def test_employment_placeholder_becomes_missing_with_flag():
    df = clean_application(_app())
    assert df["EMPLOYED_ANOMALY"].tolist() == [0, 1, 0, 0]
    assert df["YEARS_EMPLOYED"].isna().tolist() == [False, True, False, False]


def test_gender_xna_is_missing():
    assert clean_application(_app())["CODE_GENDER"].isna().tolist() == [False, False, True, False]


def test_income_cap_is_reused_on_new_data():
    train = clean_application(_app())
    new = clean_application(_app(), income_cap=train.attrs["income_cap"])
    assert new["AMT_INCOME_TOTAL"].max() <= train.attrs["income_cap"]


def test_ages_are_positive_years():
    df = clean_application(_app())
    assert (df["AGE_YEARS"] > 0).all()
    assert "DAYS_BIRTH" not in df.columns


def test_ratios_never_divide_by_zero():
    df = add_application_features(clean_application(_app()))
    ratio_cols = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_ANNUITY_RATIO", "INCOME_PER_PERSON"]
    assert not np.isinf(df[ratio_cols].to_numpy(dtype=float)).any()
    assert df.loc[2, "EXT_SOURCE_N_MISSING"] == 3


@pytest.fixture
def many_applicants():
    rng = np.random.default_rng(0)
    n = 10_000
    return pd.DataFrame({
        ID_COL: np.arange(n),
        TARGET_COL: (rng.random(n) < 0.08).astype(int),
        "THIN_FILE": (rng.random(n) < 0.14).astype(int),
    })


def test_splits_cover_everyone_once(many_applicants):
    splits = make_splits(many_applicants)
    assert len(splits) == len(many_applicants)
    assert splits[ID_COL].is_unique
    assert set(splits["split"]) == set(SPLIT_FRACTIONS)


def test_splits_are_stratified(many_applicants):
    df = many_applicants.merge(make_splits(many_applicants), on=ID_COL)
    by_split = df.groupby("split")[[TARGET_COL, "THIN_FILE"]].mean()
    overall = df[[TARGET_COL, "THIN_FILE"]].mean()
    assert (by_split - overall).abs().max().max() < 0.005
    shares = df["split"].value_counts(normalize=True)
    for name, frac in SPLIT_FRACTIONS.items():
        assert shares[name] == pytest.approx(frac, abs=0.005)


def test_splits_are_reproducible(many_applicants):
    a = make_splits(many_applicants).sort_values(ID_COL).reset_index(drop=True)
    b = make_splits(many_applicants).sort_values(ID_COL).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_protected_attributes_never_model_inputs():
    df = add_application_features(clean_application(_app()))
    df["split"] = "train"
    cols = model_columns(df)
    for col in PROTECTED_COLS + EXCLUDED_INPUTS + [ID_COL, TARGET_COL, "split"]:
        assert col not in cols
