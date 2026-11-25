"""Tests for borrower personas: clustering pipeline, naming rules and the saved bundle."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secondlook import personas as PE  # noqa: E402

BUNDLE = ROOT / "app" / "demo_data" / "personas.parquet"


def blobs(n=60, seed=0):
    """Three well-separated groups on the persona features, with a few missing values."""
    rng = np.random.default_rng(seed)
    centres = [0, 5, 10]
    rows = [{f: c + rng.normal(scale=0.3) for f in PE.PERSONA_FEATURES} for c in centres for _ in range(n)]
    X = pd.DataFrame(rows)
    X.iloc[0, 0] = np.nan
    return X


def test_pipeline_imputes_scales_and_finds_clear_groups():
    scan = PE.scan_k(blobs(), ks=[2, 3, 4])
    assert scan["silhouette"].idxmax() == 3                    # the true number of groups wins
    labels = PE.make_pipeline(3).fit(blobs())[-1].labels_
    assert sorted(np.bincount(labels)) == [60, 60, 60]


def test_scaling_stops_one_large_feature_deciding():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({f: rng.normal(size=200) for f in PE.PERSONA_FEATURES})
    X["EXT_SOURCE_MEAN"] = np.r_[np.zeros(100), np.ones(100) * 3]      # the real split, on a small scale
    X["BUR_N_LOANS"] = rng.normal(scale=1000, size=200)                # noise on a huge scale
    labels = PE.make_pipeline(2).fit(X)[-1].labels_
    # Scaled, both the split and the noise count equally: the split is still mostly recovered
    agree = max((labels[:100] == 0).mean() + (labels[100:] == 1).mean(), (labels[:100] == 1).mean() + (labels[100:] == 0).mean()) / 2
    assert agree > 0.7


def test_names_come_from_distinguishing_traits():
    prof = pd.DataFrame({
        "HAS_BUREAU_HISTORY": [1.0, 0.0, 1.0, 0.9, 1.0],
        "INST_LATE_SHARE": [0.03, 0.08, 0.33, 0.05, 0.07],
        "CREDIT_TO_INCOME": [2.7, 3.2, 3.0, 7.1, 3.2],
        "YEARS_EMPLOYED": [2.3, 2.6, 2.7, 3.0, 9.4],
    }, index=[10, 11, 12, 13, 14])
    assert PE.name_personas(prof) == {11: "Thin file", 12: "Missed payments before", 13: "Big loan for the income",
                                      14: "Long job, many past loans", 10: "Newer job, clean record"}


def test_cramers_v_bounds():
    a = pd.Series(list("xy") * 50)
    assert PE.cramers_v(a, a) == pytest.approx(1.0)
    assert PE.cramers_v(a, pd.Series(list("xxyy") * 25)) == pytest.approx(0.0, abs=1e-9)


def test_saved_bundle_has_five_named_personas():
    p = pd.read_parquet(BUNDLE)
    assert set(p["persona"]) == {"Newer job, clean record", "Thin file", "Missed payments before",
                                 "Big loan for the income", "Long job, many past loans"}
    assert len(p) == 7263 and p["SK_ID_CURR"].is_unique
    assert (p["PD"] >= 0.17).all()                                     # only applicants the model does not approve
    assert not {"AGE_YEARS", "CODE_GENDER"} & set(p.columns)          # sensitive traits are not in the bundle
    prof = PE.profile(p)
    assert prof["Repayment rate"].between(0.6, 0.8).all()              # the notebook's finding: similar risk in every persona
    assert all(len(PE.describe(n, r)) > 50 for n, r in prof.iterrows())
