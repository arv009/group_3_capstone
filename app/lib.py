"""Shared loaders and helpers for the SecondLook Streamlit app."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secondlook import modeling as M                      # noqa: E402
from secondlook.explain import Explainer, label_of        # noqa: E402
from secondlook import letters as L                       # noqa: E402
from secondlook import path as P                          # noqa: E402

BUNDLE = ROOT / "app" / "demo_data"
DECISION_STYLE = {
    "approve": ("Approve", "#2E7D32", ""),
    "under_review": ("Under review", "#E08E0B", ""),
    "decline": ("Decline", "#C62828", ""),
}


@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    import lightgbm as lgb
    # Load from a string with normalised line endings: a CRLF copy of the file (e.g. a Windows checkout) crashes LightGBM
    model_str = (BUNDLE / "model.txt").read_bytes().decode("utf-8").replace("\r\n", "\n")
    booster = lgb.Booster(model_str=model_str)
    meta = json.loads((BUNDLE / "model_meta.json").read_text())
    return booster, meta, Explainer(booster)


@st.cache_data(show_spinner=False)
def load_portfolio() -> pd.DataFrame:
    return pd.read_parquet(BUNDLE / "portfolio.parquet")


@st.cache_data(show_spinner=False)
def load_applicants() -> pd.DataFrame:
    return pd.read_parquet(BUNDLE / "applicants.parquet")


@st.cache_data(show_spinner=False)
def load_letters() -> dict:
    path = BUNDLE / "letters.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_personas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persona per not-approved test applicant, and one profile row per persona (notebook 09)."""
    return pd.read_parquet(BUNDLE / "personas.parquet"), pd.read_json(BUNDLE / "personas_profile.json")


@st.cache_data(show_spinner=False)
def load_fairness() -> dict:
    """Fairness audit results: group metrics, bootstrap gaps and the retraining experiment (notebook 08)."""
    return json.loads((BUNDLE / "fairness.json").read_text())


@st.cache_data(show_spinner=False)
def load_letter_audits() -> dict:
    """Faithfulness and tone audits of the pre-drafted letters (notebook 07), keyed like letters.json."""
    path = BUNDLE / "letter_audits.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def audit_live_letter(letter: dict) -> dict:
    """Audit a letter drafted live in the app (two to four Claude Haiku 5.5 calls, about $0.0004)."""
    import anthropic
    from secondlook.letter_audit import AuditCache, audit_letter
    a = audit_letter(anthropic.Anthropic(), letter["body"], letter["fact_sheet"], AuditCache(path=None))
    return a.to_dict()


@st.cache_data(show_spinner="Scoring applicants...")
def score_applicants() -> pd.DataFrame:
    """PD and decision for every applicant in the bundle."""
    booster, meta, _ = load_model()
    apps = load_applicants()
    X = M.prepare_X(apps, meta["features"], meta["categorical"], meta["categories"])
    out = apps[["SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_TERM", "HAS_BUREAU_HISTORY"]].copy()
    out["PD"] = booster.predict(X)
    out["DECISION"] = out["PD"].map(lambda p: L.decision_for(p, meta["cutoff"]))
    return out


def applicant_X(sid: int) -> pd.DataFrame:
    _, meta, _ = load_model()
    apps = load_applicants()
    row = apps[apps["SK_ID_CURR"] == sid]
    return M.prepare_X(row, meta["features"], meta["categorical"], meta["categories"])


@st.cache_data(show_spinner=False)
def explain_applicant(sid: int) -> dict:
    _, _, explainer = load_model()
    return explainer.explain(applicant_X(sid))


@st.cache_data(show_spinner=False)
def path_for(sid: int) -> dict:
    """Re-scored path to approval for one applicant (see src/secondlook/path.py)."""
    booster, meta, _ = load_model()
    apps = load_applicants()
    return P.path_to_approval(apps[apps["SK_ID_CURR"] == sid], booster, meta, meta["cutoff"])


def api_key_available() -> bool:
    """True if a Claude API key is set (Streamlit secrets in the cloud, or .env locally)."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            os.environ.setdefault("ANTHROPIC_API_KEY", st.secrets["ANTHROPIC_API_KEY"])
    except Exception:
        pass                     # no secrets file locally: fine
    if not os.getenv("ANTHROPIC_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def decision_badge(decision: str) -> str:
    text, color, _ = DECISION_STYLE[decision]
    return (f"<span style='background:{color};color:white;padding:4px 12px;border-radius:12px;"
            f"font-weight:600'>{text}</span>")


def money(x: float) -> str:
    return f"{x:,.0f}"


def approvals_at_rate(y, score, rate):
    return M.approvals_at_default_rate(np.asarray(y), np.asarray(score), rate)[0]


def footer():
    st.divider()
    st.caption("SecondLook · Group 3 capstone, Uplift Code Camp · Data: Home Credit Default Risk (Kaggle). "
               "Demo on held-out test applicants; profit figures use stated assumptions, not real lender economics.")
