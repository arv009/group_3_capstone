"""Tests for the path to approval (uses the small app bundle, so no Kaggle data is needed)."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secondlook import modeling as M  # noqa: E402
from secondlook.letters import decision_for  # noqa: E402
from secondlook.path import LOAN_FEATURES, MARGIN, TERM_CAP, apply_terms, path_to_approval  # noqa: E402

BUNDLE = ROOT / "app" / "demo_data"


@pytest.fixture(scope="module")
def setup():
    booster = lgb.Booster(model_str=(BUNDLE / "model.txt").read_bytes().decode("utf-8").replace("\r\n", "\n"))
    meta = json.loads((BUNDLE / "model_meta.json").read_text())
    apps = pd.read_parquet(BUNDLE / "applicants.parquet")
    X = M.prepare_X(apps, meta["features"], meta["categorical"], meta["categories"])
    apps["PD"] = booster.predict(X)
    apps["DECISION"] = apps["PD"].map(lambda p: decision_for(p, meta["cutoff"]))
    return booster, meta, apps


def one(apps, decision, i=0):
    return apps[apps["DECISION"] == decision].iloc[[i]].drop(columns=["PD", "DECISION"])


def test_unchanged_terms_give_the_same_pd(setup):
    booster, meta, apps = setup
    row = one(apps, "under_review")
    res = path_to_approval(row, booster, meta, meta["cutoff"])
    assert res["current"]["pd"] == pytest.approx(apps[apps["SK_ID_CURR"] == row["SK_ID_CURR"].iloc[0]]["PD"].iloc[0], abs=1e-12)


def test_approved_applicant_needs_no_change(setup):
    booster, meta, apps = setup
    res = path_to_approval(one(apps, "approve"), booster, meta, meta["cutoff"])
    assert res["status"] == "approved" and res["recommended"] is None


def test_every_option_is_under_the_cutoff_and_never_a_bigger_loan(setup):
    booster, meta, apps = setup
    for i in range(5):
        row = one(apps, "under_review", i)
        res = path_to_approval(row, booster, meta, meta["cutoff"])
        if res["status"] == "path":
            assert (res["options"]["pd"] < meta["cutoff"]).all()
            assert (res["options"]["loan_amount"] <= row["AMT_CREDIT"].iloc[0] + 1e-6).all()
            assert (res["options"]["n_payments"] <= max(TERM_CAP, row["CREDIT_TERM"].iloc[0]) + 1e-6).all()


def test_only_loan_features_change(setup):
    _, _, apps = setup
    row = one(apps, "decline")
    changed = apply_terms(row, 0.2, 1.3)
    diff = [c for c in row.columns if not (row[c].isna().all() and changed[c].isna().all()) and not (row[c].values == changed[c].values).all()]
    assert set(diff) <= set(LOAN_FEATURES)
    assert changed["AMT_CREDIT"].iloc[0] == pytest.approx(0.8 * row["AMT_CREDIT"].iloc[0])
    assert changed["AMT_ANNUITY"].iloc[0] == pytest.approx(changed["AMT_CREDIT"].iloc[0] / changed["CREDIT_TERM"].iloc[0])


def test_revolving_loans_keep_their_term(setup):
    booster, meta, apps = setup
    rev = apps[(apps["NAME_CONTRACT_TYPE"] == "Revolving loans") & (apps["DECISION"] != "approve")]
    if rev.empty:
        pytest.skip("no revolving loan above the cutoff in the bundle")
    res = path_to_approval(rev.iloc[[0]].drop(columns=["PD", "DECISION"]), booster, meta, meta["cutoff"])
    assert (res["options"]["term_mult"] == 1.0).all()


def test_recommended_is_the_smallest_change_with_a_safety_margin(setup):
    booster, meta, apps = setup
    for i in range(5):
        res = path_to_approval(one(apps, "under_review", i), booster, meta, meta["cutoff"])
        if res["status"] == "path":
            opts = res["options"]
            assert res["recommended"]["change"] == pytest.approx(opts["change"].min())
            tied = opts[opts["change"] == opts["change"].min()]
            assert res["recommended"]["expected_profit"] == pytest.approx(np.max(tied["expected_profit"]))
            assert (opts["pd"] < meta["cutoff"] - MARGIN).all()
