"""Unit tests for the decision and profit helpers in modeling.py (no Kaggle data needed)."""
import numpy as np
import pandas as pd
import pytest

from secondlook.modeling import (
    ALT_FLAGS, bootstrap_auc_diff, ProfitAssumptions, approval_summary, cutoff_for_bad_rate, feature_sets, fit_calibrator, profit_curve,
)


def test_cutoff_keeps_approved_bad_rate_within_target():
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = (rng.random(5000) < p * 0.3).astype(int)  # riskier scores default more
    cutoff = cutoff_for_bad_rate(y, p, 0.05)
    assert y[p <= cutoff].mean() <= 0.05
    # approving one more applicant (the next riskiest) must break the target or be impossible
    nxt = np.sort(p[p > cutoff])
    if len(nxt):
        assert y[p <= nxt[0]].mean() > 0.05 or nxt[0] == cutoff


def test_cutoff_when_target_impossible_approves_nobody():
    y = np.array([1, 1, 1])
    p = np.array([0.1, 0.2, 0.3])
    cutoff = cutoff_for_bad_rate(y, p, 0.05)
    assert (p <= cutoff).sum() == 0


def test_approval_summary_counts():
    y = np.array([0, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.9])
    thin = np.array([True, False, True, True, False])
    s = approval_summary(y, p, cutoff=0.3, thin=thin)
    assert s["approval_rate"] == pytest.approx(0.6)
    assert s["bad_rate_approved"] == pytest.approx(1 / 3)
    assert s["thin_approval_rate"] == pytest.approx(2 / 3)
    assert s["thin_good_approved"] == 1 and s["thin_good_total"] == 2


def test_profit_curve_economics():
    econ = ProfitAssumptions(margin=0.1, lgd=0.5)
    y = np.array([0, 1])
    amount = np.array([1000.0, 1000.0])
    curve = profit_curve(y, np.array([0.1, 0.2]), amount, econ, cutoffs=[0.0, 0.15, 0.25]).set_index("cutoff")
    assert curve.loc[0.0, "profit"] == 0
    assert curve.loc[0.15, "profit"] == pytest.approx(100)        # repaid loan only
    assert curve.loc[0.25, "profit"] == pytest.approx(100 - 500)  # plus the default
    assert econ.breakeven_pd == pytest.approx(0.1 / 0.6)


def test_calibrator_is_monotonic_and_bounded():
    rng = np.random.default_rng(2)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    cal = fit_calibrator(y, p)
    out = cal.predict(np.linspace(-1, 2, 50))
    assert (np.diff(out) >= 0).all() and out.min() >= 0 and out.max() <= 1


def test_conventional_model_has_no_alternative_data():
    df = pd.DataFrame(columns=["SK_ID_CURR", "TARGET", "split", "AMT_CREDIT", "BUREAU_N_LOANS",
                               "INST_LATE_SHARE", "PREV_N_APPS", "POS_N_MONTHS", "CC_N_MONTHS", *ALT_FLAGS])
    sets = feature_sets(df)
    assert sets["conventional"] == ["AMT_CREDIT", "BUREAU_N_LOANS"]
    assert set(sets["conventional"]) < set(sets["secondlook"])


def test_bootstrap_detects_a_real_gain_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    weak = y + rng.normal(0, 2.0, 3000)
    strong = y + rng.normal(0, 0.5, 3000)
    seg = {"all": np.ones(3000, bool)}
    assert bootstrap_auc_diff(y, weak, strong, seg, n_boot=100).loc["all", "significant"]
    assert not bootstrap_auc_diff(y, weak, weak + 1e-9, seg, n_boot=100).loc["all", "significant"]
