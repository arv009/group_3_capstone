"""Unit tests for conformal.py and fairness.py (synthetic data, no Kaggle files needed)."""
import numpy as np
import pandas as pd
import pytest

from secondlook.conformal import (assign_zones, bin_edges, binned_intervals, class_conditional_thresholds, decide,
                                  marginal_threshold, prediction_sets, set_decision)
from secondlook.fairness import (apply_thresholds, blend_thresholds, equal_opportunity_gap, group_report,
                                 parity_thresholds, portfolio_profit)


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.beta(1.2, 10, n)                      # true default probabilities
    y = (rng.random(n) < p).astype(int)
    score = np.clip(p + rng.normal(0, 0.03, n), 0, 1)  # an imperfect model score
    return y, score


def test_class_conditional_coverage_holds_for_both_outcomes():
    y_cal, s_cal = synthetic(20000, 0)
    y_new, s_new = synthetic(20000, 1)
    q0, q1 = class_conditional_thresholds(y_cal, s_cal, alpha=0.1)
    has_repay, has_default = prediction_sets(s_new, q0, q1)
    assert has_repay[y_new == 0].mean() >= 0.88   # 90% target, small sampling slack
    assert has_default[y_new == 1].mean() >= 0.88


def test_marginal_conformal_covers_on_average_only():
    y_cal, s_cal = synthetic(20000, 2)
    y_new, s_new = synthetic(20000, 3)
    q = marginal_threshold(y_cal, s_cal, alpha=0.1)
    covered = np.where(y_new == 1, 1 - s_new, s_new) <= q
    assert covered.mean() >= 0.88
    assert covered[y_new == 1].mean() < covered[y_new == 0].mean()  # the imbalance trap


def test_conformal_quantile_with_too_few_points_is_infinite():
    q0, _ = class_conditional_thresholds(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9]), alpha=0.1)
    assert q0 == np.inf


def test_set_decision_mapping():
    out = set_decision([True, False, True, False], [False, True, True, False])
    assert out.tolist() == ["approve", "decline", "refer", "refer"]


def test_binned_intervals_cover_true_rates_and_zones_are_monotone():
    y_ref, s_ref = synthetic(20000, 4)
    y_cal, s_cal = synthetic(30000, 5)
    edges = bin_edges(s_ref, 10)
    iv = binned_intervals(y_cal, s_cal, edges, confidence=0.95)
    assert iv["n"].sum() == len(y_cal)
    assert ((iv["low"] <= iv["default_rate"]) & (iv["default_rate"] <= iv["high"])).all()
    zones = assign_zones(iv, breakeven_pd=0.12)
    order = {"approve": 0, "refer": 1, "decline": 2}
    assert np.all(np.diff([order[z] for z in zones["zone"]]) >= 0)
    d = decide(np.array([-1.0, 0.0, 2.0]), zones)
    assert d[0] == zones["zone"].iloc[0] and d[-1] == zones["zone"].iloc[-1]


def test_group_report_and_gap():
    y = np.array([0, 0, 1, 0, 0, 1])
    approve = np.array([1, 1, 0, 1, 0, 0])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    rep = group_report(y, approve, g)
    assert rep.loc["a", "good-borrower approval"] == 1.0
    assert rep.loc["b", "good-borrower approval"] == 0.5
    assert rep.loc["overall", "applicants"] == 6
    assert equal_opportunity_gap(y, approve, g) == pytest.approx(0.5)


def test_parity_thresholds_equalise_good_borrower_approval():
    rng = np.random.default_rng(6)
    n = 40000
    g = rng.choice(["young", "old"], n)
    p = np.clip(rng.beta(2, 15, n) + np.where(g == "young", 0.05, 0.0), 0, 1)  # young score riskier
    y = (rng.random(n) < p).astype(int)
    cutoff = 0.15
    single = p <= cutoff
    th = parity_thresholds(y, p, g, cutoff)
    fair = apply_thresholds(p, g, th, cutoff)
    assert equal_opportunity_gap(y, single, g) > 0.1
    assert equal_opportunity_gap(y, fair, g) < 0.01
    half = apply_thresholds(p, g, blend_thresholds(th, cutoff, 0.5), cutoff)
    assert equal_opportunity_gap(y, fair, g) < equal_opportunity_gap(y, half, g) < equal_opportunity_gap(y, single, g)


def test_portfolio_profit():
    assert portfolio_profit([0, 1], [1, 1], [100, 100], margin=0.1, lgd=0.5) == pytest.approx(10 - 50)
    assert portfolio_profit([0, 1], [1, 0.5], [100, 100], margin=0.1, lgd=0.5) == pytest.approx(10 - 25)
