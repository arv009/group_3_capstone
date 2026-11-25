"""Tests for the fairness audit metrics on a small hand-made portfolio where every rate is known."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secondlook import fairness as F  # noqa: E402


def portfolio(reps=100):
    # Group A: 10 applicants, 1 defaulter; 9 approved (PD 0.05), the defaulter and nobody else declined.
    # Group B: 10 applicants, 2 defaulters; 6 approved, including 1 defaulter; 2 repayers declined.
    a = pd.DataFrame({"G": "A", "TARGET": [1] + [0] * 9, "PD": [0.30] + [0.05] * 9})
    b = pd.DataFrame({"G": "B", "TARGET": [1, 1] + [0] * 8, "PD": [0.30, 0.05] + [0.05] * 5 + [0.30] * 3})
    return pd.concat([a, b] * reps, ignore_index=True)


def test_group_table_rates():
    t = F.group_table(portfolio(), "G", pd_col="PD", cutoff=0.17)
    assert t.loc["A", "Approval rate"] == pytest.approx(0.9)
    assert t.loc["B", "Approval rate"] == pytest.approx(0.6)
    assert t.loc["A", "Approval rate of repayers"] == pytest.approx(1.0)
    assert t.loc["B", "Decline rate of repayers"] == pytest.approx(3 / 8)
    assert t.loc["B", "Default rate of approved"] == pytest.approx(1 / 6)
    assert t.loc["B", "Four-fifths ratio"] == pytest.approx(0.6 / 0.9)
    assert bool(t.loc["B", "Below 0.8"]) and not bool(t.loc["A", "Below 0.8"])
    assert t.loc["A", "Calibration gap"] == pytest.approx((0.30 + 9 * 0.05) / 10 - 0.1)


def test_small_groups_are_dropped():
    df = pd.concat([portfolio(), pd.DataFrame({"G": ["C"] * 5, "TARGET": 0, "PD": 0.05})])
    assert "C" not in F.group_table(df, "G", pd_col="PD").index


def test_bootstrap_point_estimates_and_intervals():
    gaps = F.bootstrap_gaps(portfolio(), "G", pd_col="PD", n_boot=200).set_index("Metric")
    assert gaps.loc["Approval rate", "Gap"] == pytest.approx(0.3)
    assert gaps.loc["Approval rate of repayers", "Gap"] == pytest.approx(1 - 5 / 8)
    assert gaps.loc["Four-fifths ratio", "Gap"] == pytest.approx(0.6 / 0.9)
    assert (gaps["95% CI low"] <= gaps["Gap"]).all() and (gaps["Gap"] <= gaps["95% CI high"]).all()
    assert gaps.loc["Approval rate", "Real (CI excludes no gap)"]


def test_bootstrap_finds_no_gap_between_identical_groups():
    df = portfolio()
    df["G"] = np.where((np.arange(len(df)) // 20) % 2 == 0, "X", "Y")    # whole A+B blocks: same mix in both
    gaps = F.bootstrap_gaps(df, "G", pd_col="PD", n_boot=300).set_index("Metric")
    assert not gaps.loc["Approval rate", "Real (CI excludes no gap)"]


def test_age_bands_and_labels():
    df = pd.DataFrame({"CODE_GENDER": ["F", "M", None], "AGE_YEARS": [29.99, 30.0, 68.9], "HAS_BUREAU_HISTORY": [1, 0, 1]})
    g = F.add_groups(df)
    assert g["Age band"].astype(str).tolist() == ["20-29", "30-39", "60-69"]
    assert g["Gender"].tolist()[:2] == ["Female", "Male"] and pd.isna(g["Gender"].iloc[2])
    assert g["Credit file"].tolist() == ["Bureau history", "Thin file", "Bureau history"]
