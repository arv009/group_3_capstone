"""Train, calibrate and evaluate the default-risk models (notebook 03).

Three models are compared on the same fixed splits:
  * Logistic regression: simple, transparent baseline.
  * Conventional LightGBM: application + credit bureau data only, i.e. what a typical
    lender sees. This is the benchmark SecondLook has to beat on thin-file applicants.
  * SecondLook LightGBM: adds alternative data (previous applications, repayment history,
    POS and card behaviour).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODELS_DIR, RANDOM_STATE, TARGET_COL
from .features import model_columns

# Alternative-data features: everything a conventional bureau-based lender would NOT use.
ALT_PREFIXES = ("PREV_", "INST_", "POS_", "CC_")
ALT_FLAGS = ("HAS_PREV_APP", "HAS_REPAYMENT_HISTORY", "PRIOR_REFUSAL")

LGBM_PARAMS = dict(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    verbose=-1,
)


# ----------------------------------------------------------------------------- data
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Text columns -> pandas category with the SAME categories in every split.
    Call once on the full table before splitting."""
    df = df.copy()
    for col in model_columns(df):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")
    return df


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Column lists for the conventional and SecondLook models."""
    full = model_columns(df)
    conventional = [c for c in full if not c.startswith(ALT_PREFIXES) and c not in ALT_FLAGS]
    return {"conventional": conventional, "secondlook": full}


def split_xy(df: pd.DataFrame, split: str, cols: list[str]):
    part = df[df["split"] == split]
    return part[cols], part[TARGET_COL].to_numpy()


# ----------------------------------------------------------------------------- models
def train_logistic(X: pd.DataFrame, y) -> Pipeline:
    """Median-impute + scale numbers, one-hot encode categories, L2 logistic regression."""
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)), cat),
    ])
    X = X.copy()
    X[cat] = X[cat].astype(str)
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.1, max_iter=2000))])
    return model.fit(X, y)


def predict_logistic(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    X = X.copy()
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X[cat] = X[cat].astype(str)
    return model.predict_proba(X)[:, 1]


def train_lightgbm(X_train, y_train, X_valid, y_valid, params: dict | None = None,
                   sample_weight=None) -> lgb.LGBMClassifier:
    """LightGBM with early stopping on the validation split (AUC). Optional per-row weights
    (used by reject inference, where each inferred applicant counts fractionally)."""
    model = lgb.LGBMClassifier(**(params or LGBM_PARAMS))
    with warnings.catch_warnings():
        # LightGBM >= 4.7 renames eval_set to eval_X/eval_y; eval_set still works and
        # keeps us compatible with older versions teammates may have installed.
        warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
        model.fit(
            X_train, y_train, sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)], eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
    return model


# ----------------------------------------------------------------------------- evaluation
def evaluate(y, p, thin=None) -> dict:
    """ROC-AUC, PR-AUC and Brier score; also for the thin-file segment if a mask is given."""
    out = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}
    if thin is not None:
        thin = np.asarray(thin, dtype=bool)
        out |= {"thin_roc_auc": roc_auc_score(y[thin], p[thin]), "thin_pr_auc": average_precision_score(y[thin], p[thin])}
    return out


def bootstrap_auc_diff(y, p_base, p_new, segments: dict, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap of AUC(p_new) - AUC(p_base): resample applicants with replacement, score both
    models on the same resample, per segment ({name: boolean mask}). Returns mean and 95% interval;
    'significant' means the interval excludes zero."""
    y, p_base, p_new = np.asarray(y), np.asarray(p_base), np.asarray(p_new)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for name, mask in segments.items():
            i = idx[np.asarray(mask)[idx]]
            rows.append({"segment": name, "diff": roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_base[i])})
    out = pd.DataFrame(rows).groupby("segment", sort=False)["diff"].agg(
        mean="mean", low=lambda d: d.quantile(0.025), high=lambda d: d.quantile(0.975))
    out["significant"] = (out["low"] > 0) | (out["high"] < 0)
    return out


def fit_calibrator(y, p) -> IsotonicRegression:
    """Isotonic calibration: maps raw scores to observed default rates. Fit on the calib split only."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)


def cutoff_for_bad_rate(y, p, target_bad_rate: float) -> float:
    """Highest risk cutoff such that applicants approved (p <= cutoff) default at most target_bad_rate.

    Approving in order of lowest predicted risk, find the largest group whose observed
    default rate stays within the target.
    """
    order = np.argsort(p, kind="stable")
    y_sorted, p_sorted = np.asarray(y)[order], np.asarray(p)[order]
    cum_bad_rate = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    ok = np.nonzero(cum_bad_rate <= target_bad_rate)[0]
    if len(ok) == 0:
        return float(p_sorted[0]) - 1e-12
    return float(p_sorted[ok[-1]])


def approval_summary(y, p, cutoff: float, thin) -> dict:
    """What happens if we approve everyone with p <= cutoff."""
    y, thin = np.asarray(y), np.asarray(thin, dtype=bool)
    approve = np.asarray(p) <= cutoff
    return {
        "approval_rate": approve.mean(),
        "bad_rate_approved": y[approve].mean() if approve.any() else np.nan,
        "thin_approval_rate": approve[thin].mean(),
        "thin_bad_rate_approved": y[approve & thin].mean() if (approve & thin).any() else np.nan,
        "thin_good_approved": int((approve & thin & (y == 0)).sum()),
        "thin_good_total": int((thin & (y == 0)).sum()),
    }


@dataclass
class ProfitAssumptions:
    """Simple unit economics per loan, as a share of the credit amount.

    margin: net interest + fees earned over the loan's life if it is repaid.
    lgd: loss given default, the share of the credit lost if the borrower defaults.
    Break-even probability of default = margin / (margin + lgd).
    """
    margin: float = 0.10
    lgd: float = 0.65
    notes: str = field(default="Illustrative assumptions; change them to test sensitivity.")

    @property
    def breakeven_pd(self) -> float:
        return self.margin / (self.margin + self.lgd)


def profit_curve(y, p, amount, assumptions: ProfitAssumptions = ProfitAssumptions(), cutoffs=None) -> pd.DataFrame:
    """Realised profit, approval rate and bad rate when approving p <= cutoff, for each cutoff."""
    y, p, amount = np.asarray(y), np.asarray(p), np.asarray(amount, dtype=float)
    per_loan = np.where(y == 0, assumptions.margin * amount, -assumptions.lgd * amount)
    cutoffs = np.linspace(0.0, 0.5, 201) if cutoffs is None else cutoffs
    rows = []
    for c in cutoffs:
        a = p <= c
        rows.append({"cutoff": c, "approval_rate": a.mean(), "bad_rate": y[a].mean() if a.any() else np.nan,
                     "profit": per_loan[a].sum()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- persistence
def save_bundle(name: str, model, calibrator, columns: list[str], categories: dict, meta: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "columns": columns, "categories": categories},
                MODELS_DIR / f"{name}.joblib")
    (MODELS_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, default=float))


def load_bundle(name: str) -> dict:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebooks/03_modelling.ipynb first.")
    return joblib.load(path)
