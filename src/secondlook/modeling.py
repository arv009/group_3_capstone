"""Train, evaluate and save the SecondLook risk models.

Two models are compared:
  * a logistic regression baseline (simple, interpretable, needs imputation + scaling)
  * LightGBM, a gradient-boosted tree model (handles missing values, non-linear patterns
    and interactions on its own)

The model predicts PD, the probability of default (TARGET = 1). Lending decisions are
then made by comparing PD with a cutoff chosen for profit, not for accuracy.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import ID_COL, MODELS_DIR, RANDOM_STATE, TARGET_COL

PROTECTED_COLS = ["CODE_GENDER"]          # fairness audit only, never a model input
TEST_SIZE = 0.20

LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",              # early stopping watches AUC only (not log-loss)
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,     # each leaf needs 100+ applicants: guards against overfitting
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.5,      # each tree sees half the features: more diverse trees
    reg_lambda=5.0,
    n_estimators=5000,         # upper bound; early stopping picks the real number
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def model_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (all model features, categorical features). Excludes ID, target, protected columns."""
    feats = [c for c in df.columns if c not in (ID_COL, TARGET_COL, *PROTECTED_COLS)]
    cats = [c for c in feats if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    return feats, cats


def prepare_X(df: pd.DataFrame, feats: list[str], cats: list[str], categories: dict | None = None) -> pd.DataFrame:
    """Select features and give categorical columns a fixed category list (same codes in train and app)."""
    X = df[feats].copy()
    for c in cats:
        cat_type = pd.CategoricalDtype(categories[c]) if categories else None
        X[c] = X[c].astype(cat_type or "category")
    return X


def split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Stratified train/test split: both parts keep the same 8% default rate."""
    return train_test_split(df, test_size=test_size, stratify=df[TARGET_COL], random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Baseline: logistic regression
# ---------------------------------------------------------------------------
_MONEY_PREFIXES = ("AMT_", "BUR_TOTAL_", "INST_TOTAL_", "CC_AVG_BALANCE", "CC_AVG_DRAWINGS",
                   "PREV_AVG_AMT_", "INCOME_PER_PERSON")


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_logistic(num_cols: list[str], cat_cols: list[str], class_weight=None) -> Pipeline:
    """Impute -> (log money columns) -> scale -> one-hot categories -> logistic regression."""
    money = [c for c in num_cols if c.startswith(_MONEY_PREFIXES)]
    other = [c for c in num_cols if c not in money]
    pre = ColumnTransformer([
        ("money", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(_signed_log),
                                StandardScaler()), money),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), other),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="Missing"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200)), cat_cols),
    ])
    clf = LogisticRegression(C=0.1, max_iter=3000, class_weight=class_weight)
    return Pipeline([("prep", pre), ("model", clf)])


# ---------------------------------------------------------------------------
# Main model: LightGBM with cross-validation
# ---------------------------------------------------------------------------
def cross_validate_lgbm(X: pd.DataFrame, y: pd.Series, params: dict | None = None, n_splits: int = 5,
                        verbose: bool = True) -> dict:
    """Stratified K-fold CV with early stopping. Returns fold AUCs, best rounds and out-of-fold predictions."""
    params = {**LGBM_PARAMS, **(params or {})}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    aucs, rounds = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], oof[va]))
        rounds.append(m.best_iteration_)
        if verbose:
            print(f"  fold {fold}: AUC {aucs[-1]:.4f}  ({rounds[-1]} trees)")
    return {"fold_auc": aucs, "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
            "best_rounds": rounds, "oof": oof}


def fit_lgbm(X: pd.DataFrame, y: pd.Series, n_estimators: int, params: dict | None = None) -> lgb.LGBMClassifier:
    """Fit the final LightGBM on all training data with a fixed number of trees (from CV)."""
    p = {**LGBM_PARAMS, **(params or {}), "n_estimators": n_estimators}
    return lgb.LGBMClassifier(**p).fit(X, y)


# ---------------------------------------------------------------------------
# Evaluation and business metrics
# ---------------------------------------------------------------------------
def scores(y, pd_hat) -> dict:
    """Ranking and probability-quality metrics."""
    return {"roc_auc": roc_auc_score(y, pd_hat),
            "pr_auc": average_precision_score(y, pd_hat),
            "brier": brier_score_loss(y, pd_hat),
            "mean_pd": float(np.mean(pd_hat)), "actual_rate": float(np.mean(y))}


@dataclass
class ProfitAssumptions:
    """Simple, explicit unit economics (stated assumptions, not Home Credit's real figures).

    margin: profit on a repaid loan, as a share of the loan amount (interest + fees - costs)
    lgd:    loss given default, the share of the loan amount lost when a borrower defaults
    """
    margin: float = 0.15
    lgd: float = 0.60


def profit_curve(y, pd_hat, amount, assumptions: ProfitAssumptions = ProfitAssumptions(),
                 cutoffs=np.round(np.arange(0.02, 0.51, 0.01), 2)) -> pd.DataFrame:
    """For each PD cutoff: approve applicants with PD < cutoff and total up the outcome."""
    y, pd_hat, amount = np.asarray(y), np.asarray(pd_hat), np.asarray(amount, dtype=float)
    rows = []
    for c in cutoffs:
        ok = pd_hat < c
        good, bad = ok & (y == 0), ok & (y == 1)
        profit = assumptions.margin * amount[good].sum() - assumptions.lgd * amount[bad].sum()
        rows.append({"cutoff": c, "approval_rate": ok.mean(), "approved": int(ok.sum()),
                     "default_rate_approved": y[ok].mean() if ok.any() else np.nan,
                     "good_declined": int(((~ok) & (y == 0)).sum()), "bad_approved": int(bad.sum()),
                     "profit": profit})
    return pd.DataFrame(rows)


def approve_all_profit(y, amount, assumptions: ProfitAssumptions = ProfitAssumptions()) -> float:
    y, amount = np.asarray(y), np.asarray(amount, dtype=float)
    return assumptions.margin * amount[y == 0].sum() - assumptions.lgd * amount[y == 1].sum()


def approvals_at_default_rate(y, score, target_rate: float) -> tuple[int, float]:
    """Approve the lowest-risk applicants first and stop when the approved group's default
    rate would exceed target_rate. Returns (number approved, PD/score cutoff)."""
    order = np.argsort(score)
    y_sorted = np.asarray(y)[order]
    running = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    ok = np.where(running <= target_rate)[0]
    n = int(ok.max() + 1) if len(ok) else 0
    return n, float(np.asarray(score)[order][n - 1]) if n else float("nan")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
MODEL_PATH = MODELS_DIR / "lgbm_model.joblib"
BASELINE_PATH = MODELS_DIR / "logistic_baseline.joblib"
META_PATH = MODELS_DIR / "model_meta.json"


def save_model(model, meta: dict, baseline=None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    if baseline is not None:
        joblib.dump(baseline, BASELINE_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))


def load_model():
    """Return (LightGBM model, metadata dict)."""
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())
