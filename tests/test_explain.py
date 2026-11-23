"""Unit tests for explain.py: lever arithmetic, consistency of derived ratios, reasons, and
the path-to-approval search (with a tiny stand-in model, so no Kaggle data is needed)."""
import numpy as np
import pandas as pd
import pytest

from secondlook.explain import DEFAULT_LEVERS, LEVERS, apply_lever, path_to_approval, top_reasons


def _applicant(**overrides):
    row = {"AMT_INCOME_TOTAL": 200_000.0, "AMT_CREDIT": 500_000.0, "AMT_ANNUITY": 25_000.0,
           "AMT_GOODS_PRICE": 450_000.0, "BUREAU_TOTAL_DEBT": 100_000.0, "BUREAU_DEBT_RATIO": 0.5,
           "CC_AVG_UTILISATION": 0.8, "CC_MAX_UTILISATION": 1.0, "CODE_GENDER": "F", "AGE_YEARS": 30.0}
    row |= overrides
    df = pd.DataFrame([row])
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    return df


@pytest.mark.parametrize("lever", [lv.name for lv in LEVERS])
def test_levers_keep_ratios_consistent_and_never_touch_protected(lever):
    out = apply_lever(_applicant(), lever, 0.2).iloc[0]
    assert out["CREDIT_INCOME_RATIO"] == pytest.approx(out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"])
    assert out["ANNUITY_INCOME_RATIO"] == pytest.approx(out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"])
    assert out["CREDIT_ANNUITY_RATIO"] == pytest.approx(out["AMT_CREDIT"] / out["AMT_ANNUITY"])
    assert out["GOODS_CREDIT_RATIO"] == pytest.approx(out["AMT_GOODS_PRICE"] / out["AMT_CREDIT"])
    assert out["CODE_GENDER"] == "F" and out["AGE_YEARS"] == 30.0
    assert out["AMT_INCOME_TOTAL"] == 200_000.0


def test_lever_arithmetic():
    a = _applicant()
    assert apply_lever(a, "borrow_less", 0.2)["AMT_GOODS_PRICE"].iloc[0] == pytest.approx(360_000)
    assert apply_lever(a, "bigger_down_payment", 0.2)["AMT_GOODS_PRICE"].iloc[0] == 450_000  # same purchase
    assert apply_lever(a, "bigger_down_payment", 0.2)["AMT_CREDIT"].iloc[0] == pytest.approx(400_000)
    assert apply_lever(a, "pay_down_debt", 0.5)["BUREAU_TOTAL_DEBT"].iloc[0] == pytest.approx(50_000)


def test_term_lever_is_opt_in():
    assert "longer_term" not in DEFAULT_LEVERS


def test_top_reasons_direction():
    contrib = pd.Series({"A": 0.5, "B": -0.3, "C": 0.1, "D": -0.6})
    x = pd.Series({"A": 1, "B": 2, "C": 3, "D": 4})
    assert top_reasons(contrib, x, n=2, direction="risk")["feature"].tolist() == ["A", "C"]
    assert top_reasons(contrib, x, n=2, direction="strength")["feature"].tolist() == ["D", "B"]


class _LoanSizeModel:
    """Stand-in model: risk rises with loan size relative to income."""
    best_iteration_ = None

    def predict_proba(self, X):
        p = np.clip(X["CREDIT_INCOME_RATIO"].to_numpy() / 10, 0, 1)
        return np.column_stack([1 - p, p])


class _Identity:
    def predict(self, p):
        return np.asarray(p)


def test_path_to_approval_finds_smallest_change():
    bundle = {"model": _LoanSizeModel(), "calibrator": _Identity(), "categories": {},
              "columns": ["CREDIT_INCOME_RATIO", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]}
    x = _applicant()  # credit/income = 2.5 -> risk 0.25
    opts = path_to_approval(bundle, x, cutoff=0.21, margin=0.0, allow_pairs=False)
    assert opts.loc[0, "current_pd"] == pytest.approx(0.25)
    assert opts.loc[0, "size"] == pytest.approx(0.20)  # 2.5 * 0.8 = 2.0 -> risk 0.20 <= 0.21
    assert (opts["new_pd"] <= 0.21).all()


def test_path_to_approval_empty_when_impossible():
    bundle = {"model": _LoanSizeModel(), "calibrator": _Identity(), "categories": {},
              "columns": ["CREDIT_INCOME_RATIO", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]}
    assert path_to_approval(bundle, _applicant(), cutoff=0.01).empty
