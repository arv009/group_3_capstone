"""Unit tests for reject_inference.py (no Kaggle data needed)."""
import numpy as np
import pandas as pd
import pytest

from secondlook.reject_inference import approval_at_bad_rate, fuzzy_augment, parcel, simulate_policy


def test_policy_declines_the_riskiest_share():
    scores = np.arange(100) / 100
    approve, cutoff = simulate_policy(scores, 0.30)
    assert approve.sum() == 70 and not approve[-30:].any()
    assert ((scores <= cutoff) == approve).all()


def test_fuzzy_augment_weights():
    X_acc = pd.DataFrame({"a": [1, 2]})
    X_rej = pd.DataFrame({"a": [3]})
    X, y, w = fuzzy_augment(X_acc, [0, 1], X_rej, [0.2], reject_weight=1.0)
    assert len(X) == 4
    assert y.tolist() == [0, 1, 1, 0]
    assert w.tolist() == pytest.approx([1, 1, 0.2, 0.8])
    # the two copies of a reject always add up to its reject weight
    assert w[2] + w[3] == pytest.approx(1.0)


def test_parcel_inflates_band_default_rate():
    rng = np.random.default_rng(0)
    n = 20_000
    p_acc = rng.random(n)
    y_acc = (rng.random(n) < p_acc * 0.2).astype(int)
    X_acc = pd.DataFrame({"a": np.zeros(n)})
    p_rej = np.full(5000, 0.95)  # all in the top band
    X_rej = pd.DataFrame({"a": np.ones(5000)})
    X, y, w = parcel(X_acc, y_acc, p_acc, X_rej, p_rej, inflation=2.0)
    top_band_rate = y_acc[p_acc >= np.quantile(p_acc, 0.9)].mean()
    assert y[n:].mean() == pytest.approx(min(top_band_rate * 2, 1), abs=0.02)
    assert (w == 1).all() and len(X) == n + 5000


def test_approval_at_bad_rate():
    y = np.array([0, 0, 0, 1, 0, 1, 1])
    p = np.arange(7) / 10
    thin = np.array([True, False, True, False, True, False, True])
    out = approval_at_bad_rate(y, p, thin, target=0.25)
    # approve first 5 (1 default in 5 = 20%); adding the 6th makes 2/6 = 33%
    assert out["approval_rate"] == pytest.approx(5 / 7)
    assert out["thin_approval_rate"] == pytest.approx(3 / 4)
