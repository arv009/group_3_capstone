"""Fairness helpers on a tiny hand-made portfolio (no data files needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secondlook.fairness import add_groups, bootstrap_gap, group_metrics  # noqa: E402


def portfolio():
    # 4 women all approved; 4 men: 2 approved. One defaulter in each group.
    return add_groups(pd.DataFrame({
        "CODE_GENDER": ["F"] * 4 + ["M"] * 4,
        "AGE_YEARS": [25, 35, 45, 65, 25, 35, 45, 19],
        "HAS_BUREAU_HISTORY": [1, 1, 0, 1, 1, 0, 1, 1],
        "TARGET": [0, 0, 0, 1, 0, 0, 0, 1],
        "PD_LGBM": [0.05, 0.10, 0.12, 0.16, 0.05, 0.30, 0.25, 0.10],
    }))


def test_group_metrics():
    m = group_metrics(portfolio(), "gender")
    assert m.loc["women", "approval_rate"] == 1.0 and m.loc["men", "approval_rate"] == 0.5
    assert m.loc["men", "repayer_approval"] == pytest.approx(1 / 3)        # 3 repayers, 1 approved
    assert m.loc["men", "ratio_to_best"] == 0.5 and m.loc["men", "below_four_fifths"]
    assert m.loc["women", "default_rate_approved"] == 0.25
    assert m.loc["women", "calibration_gap"] == pytest.approx(np.mean([0.05, 0.10, 0.12, 0.16]) - 0.25)


def test_age_bands_leave_out_under_20():
    assert portfolio()["age_band"].isna().sum() == 1


def test_bootstrap_gap_contains_point_estimate():
    df = pd.concat([portfolio()] * 50, ignore_index=True)
    r = bootstrap_gap(df, "gender", "women", "men", n_boot=200)
    assert r["gap"] == pytest.approx(1 - 1 / 3)
    assert r["low"] <= r["gap"] <= r["high"] and r["real"]
