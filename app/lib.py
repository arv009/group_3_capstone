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
from secondlook import personas as PE                     # noqa: E402

BUNDLE = ROOT / "app" / "demo_data"
STATIC = ROOT / "app" / "static"

# Brand palette (SecondLook web UI guide)
BRAND = {
    "navy": "#0A2E52", "teal": "#21A6B8", "teal_dark": "#1A8A99", "green": "#10B981", "amber": "#F59E0B",
    "red": "#EF4444", "slate": "#64748B", "slate_light": "#94A3B8", "border": "#E2E8F0", "surface": "#F4F7F9",
    "blue": "#3B82C4",
}
CATEGORICAL = [BRAND["navy"], BRAND["teal"], BRAND["amber"], BRAND["slate_light"], BRAND["blue"]]

# decision -> (label, text colour, background)
DECISION_STYLE = {
    "approve": ("Approve", "#047857", "rgba(16, 185, 129, 0.12)"),
    "under_review": ("Under review", "#B45309", "rgba(245, 158, 11, 0.14)"),
    "decline": ("Decline", "#B91C1C", "rgba(239, 68, 68, 0.12)"),
}

_CSS = """
<style>
h1, h2, h3, h4, h5, h6 { color: #0A2E52 !important; }
h1 { letter-spacing: -0.02em; line-height: 1.15 !important; }

/* metric tiles as cards */
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 16px;
  transition: box-shadow .2s ease, border-color .2s ease; }
[data-testid="stMetric"]:hover { box-shadow: 0 10px 25px -5px rgba(10, 46, 82, 0.08); }
[data-testid="stMetricLabel"] p { font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase;
  letter-spacing: 0.05em; color: #64748B !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; line-height: 1.35 !important; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; color: #0A2E52; font-weight: 600;
  font-size: clamp(1.25rem, 1.7vw, 2.1rem) !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem; }

/* markdown tables */
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
[data-testid="stMarkdownContainer"] th { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
  color: #64748B; background: #F4F7F9; }
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td { border: 1px solid #E2E8F0 !important; }
[data-testid="stMarkdownContainer"] thead:not(:has(th:not(:empty))) { display: none; }

/* buttons: teal primary with shadow, navy outline secondary */
button[data-testid="stBaseButton-primary"] { box-shadow: 0 4px 12px rgba(33, 166, 184, 0.25); border-radius: 8px; }
button[data-testid="stBaseButton-primary"]:hover { background: #1A8A99 !important; border-color: #1A8A99 !important; }
button[data-testid="stBaseButton-secondary"] { border: 1.5px solid #0A2E52 !important; color: #0A2E52 !important;
  background: transparent !important; border-radius: 8px; }
button[data-testid="stBaseButton-secondary"]:hover { border-color: #1A8A99 !important; color: #1A8A99 !important;
  background: rgba(33, 166, 184, 0.06) !important; }
button[data-testid^="stBaseButton"]:disabled { opacity: 0.4; cursor: not-allowed; box-shadow: none; }
button:focus-visible, input:focus-visible, textarea:focus-visible, a:focus-visible {
  box-shadow: 0 0 0 3px rgba(33, 166, 184, 0.3) !important; outline: none !important; }

/* page links as cards */
[data-testid="stPageLink-NavLink"] { display: flex; width: 100%; border: 1px solid #E2E8F0; border-radius: 12px; padding: 10px 14px; background: #FFFFFF;
  transition: box-shadow .2s ease, border-color .2s ease; }
[data-testid="stPageLink-NavLink"]:hover { border-color: #21A6B8; box-shadow: 0 10px 25px -5px rgba(10, 46, 82, 0.08); }
[data-testid="stPageLink-NavLink"] p { font-weight: 600; color: #0A2E52; }

[data-testid="stExpander"] details { border-radius: 12px; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #64748B; opacity: 1 !important; }

.sl-badge { display: inline-block; padding: 3px 12px; border-radius: 999px; font-family: Inter, sans-serif; font-size: 0.8rem;
  font-weight: 600; letter-spacing: 0.02em; vertical-align: middle; }
.sl-eyebrow { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin: 0; }
.sl-step { font-family: Montserrat, sans-serif; font-weight: 700; color: #21A6B8; font-size: 0.8rem; letter-spacing: 0.05em; }
</style>
"""


def page_setup(title: str | None = None):
    """Page config, logo and brand styles. Call once at the top of every page."""
    st.set_page_config(page_title=f"{title} · SecondLook" if title else "SecondLook · Second review. Smarter decisions.",
                       page_icon=str(STATIC / "favicon.png"), layout="wide")
    st.logo(str(STATIC / "logo_horizontal.png"), icon_image=str(STATIC / "logo_icon.png"), size="large")
    st.markdown(_CSS, unsafe_allow_html=True)


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
def load_audits() -> dict:
    """Independent faithfulness and tone audits of the pre-drafted letters (src/secondlook/letter_audit.py)."""
    path = BUNDLE / "audits.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_personas() -> pd.DataFrame:
    """Not-approved test applicants with their persona and PCA map position (notebooks/09_personas.ipynb)."""
    return pd.read_parquet(BUNDLE / "personas.parquet")


@st.cache_data(show_spinner=False)
def load_fairness() -> dict:
    """Fairness audit results saved by notebooks/08_fairness_audit.ipynb (tables, bootstrap gaps, fix experiment)."""
    return json.loads((BUNDLE / "fairness.json").read_text(encoding="utf-8"))


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
    text, color, bg = DECISION_STYLE[decision]
    return f"<span class='sl-badge' style='color:{color};background:{bg};border:1px solid {color}33'>{text}</span>"


def money(x: float) -> str:
    return f"{x:,.0f}"


def approvals_at_rate(y, score, rate):
    return M.approvals_at_default_rate(np.asarray(y), np.asarray(score), rate)[0]


def footer():
    st.divider()
    st.caption("SecondLook · Group 3 capstone, Uplift Code Camp · Data: Home Credit Default Risk (Kaggle). "
               "Demo on held-out test applicants; profit figures use stated assumptions, not real lender economics.")
