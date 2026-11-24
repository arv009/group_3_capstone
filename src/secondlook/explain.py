"""Explain individual risk scores with SHAP, in words a loan officer (and a letter) can use.

SHAP (SHapley Additive exPlanations) splits one prediction into a contribution per feature:

    model output (log-odds) = base value + sum of feature contributions

A positive contribution pushes the applicant's risk UP, a negative one pushes it DOWN.
For LightGBM the contributions are in log-odds; we also report the direction and a rank,
which is what reasons in a decision letter need.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .config import ROOT

# Short, plain-English names for the features most likely to appear as reasons.
LABELS = {
    "EXT_SOURCE_MEAN": "Average external credit score", "EXT_SOURCE_MIN": "Lowest external credit score",
    "EXT_SOURCE_MAX": "Highest external credit score", "EXT_SOURCE_STD": "Spread of external credit scores",
    "EXT_SOURCE_1": "External credit score 1", "EXT_SOURCE_2": "External credit score 2", "EXT_SOURCE_3": "External credit score 3",
    "ORGANIZATION_TYPE": "Type of employer", "OCCUPATION_TYPE": "Occupation",
    "CREDIT_TERM": "Loan term (number of payments)", "CREDIT_TO_GOODS": "Loan amount vs price of goods",
    "CREDIT_TO_INCOME": "Loan amount vs income", "ANNUITY_TO_INCOME": "Repayment vs income",
    "AMT_CREDIT": "Loan amount", "AMT_ANNUITY": "Regular repayment amount", "AMT_GOODS_PRICE": "Price of goods",
    "AMT_INCOME_TOTAL": "Income", "INCOME_PER_PERSON": "Income per family member",
    "YEARS_EMPLOYED": "Time in current job", "EMPLOYED_TO_AGE": "Job tenure relative to age", "AGE_YEARS": "Age",
    "NAME_EDUCATION_TYPE": "Education level", "NAME_INCOME_TYPE": "Income type", "NAME_FAMILY_STATUS": "Family status",
    "NAME_CONTRACT_TYPE": "Loan type", "FLAG_OWN_CAR": "Owns a car", "OWN_CAR_AGE": "Age of the car the applicant already owns",
    "REGION_RATING_CLIENT_W_CITY": "Region rating (with city)", "REGION_RATING_CLIENT": "Region rating",
    "YEARS_SINCE_PHONE_CHANGE": "Time since phone number changed", "YEARS_SINCE_ID_PUBLISH": "Time since ID document changed",
    "YEARS_SINCE_REGISTRATION": "Time since registration changed", "DAYS_EMPLOYED_ANOM": "Not currently employed (e.g. pensioner)",
    "BUR_DEBT_TO_CREDIT": "Debt vs credit limit at other lenders", "BUR_ACTIVE_SHARE": "Share of other loans still active",
    "BUR_YEARS_SINCE_LAST_LOAN": "Time since last loan at another lender", "BUR_YEARS_SINCE_FIRST_LOAN": "Length of credit history",
    "BUR_TOTAL_CREDIT": "Total credit at other lenders", "BUR_TOTAL_DEBT": "Total debt at other lenders",
    "BUR_N_LOANS": "Number of loans at other lenders", "BUR_N_ACTIVE": "Active loans at other lenders",
    "BUR_MAX_DAYS_OVERDUE": "Days currently overdue at other lenders", "HAS_BUREAU_HISTORY": "Has a credit bureau record",
    "BB_LATE_MONTH_SHARE": "Share of months late at other lenders",
    "PREV_AVG_GRANTED_TO_APPLIED": "Past loans granted vs amount requested", "PREV_REFUSED_SHARE": "Share of past applications refused",
    "PREV_N_REFUSED": "Past applications refused", "PREV_N_REFUSED_LAST_YEAR": "Refusals in the last year",
    "PREV_AVG_TERM": "Average term of past loans", "PREV_AVG_AMT_APPLIED": "Average amount requested before",
    "PREV_YEARS_SINCE_LAST_DECISION": "Time since last application", "PREV_N_APPROVED": "Past applications approved",
    "INST_LATE_SHARE": "Share of past payments late", "INST_LATE_SHARE_LAST_YEAR": "Share of payments late in the last year",
    "INST_AVG_DAYS_LATE": "Average days late on past payments", "INST_MAX_DAYS_LATE": "Most days late on one payment",
    "INST_UNDERPAID_SHARE": "Share of past payments underpaid", "INST_TOTAL_UNDERPAID": "Total amount underpaid",
    "POS_LATE_MONTH_SHARE": "Share of months late on store/cash loans", "POS_N_COMPLETED": "Store/cash loans completed",
    "CC_AVG_UTILISATION": "Credit card balance vs limit", "CC_AVG_BALANCE": "Average credit card balance",
    "CC_AVG_DRAWINGS": "Average credit card withdrawals", "CC_LATE_MONTH_SHARE": "Share of months late on credit card",
    "N_DOCUMENTS": "Documents provided", "BUREAU_ENQUIRIES_YEAR": "Credit enquiries in the past year",
    "EXT_SOURCE_N_MISSING": "Number of missing external scores",
    "REGION_POPULATION_RELATIVE": "Population density of home region", "POS_N_MONTHS": "Months of store/cash loan history",
    "POS_N_LOANS": "Number of past store/cash loans", "INST_N_PAYMENTS": "Number of past payments due",
    "CC_N_MONTHS": "Months of credit card history", "NAME_HOUSING_TYPE": "Housing situation",
    "WEEKDAY_APPR_PROCESS_START": "Day of week applied", "HOUR_APPR_PROCESS_START": "Hour of day applied",
    "CNT_CHILDREN": "Number of children", "CNT_FAM_MEMBERS": "Family size", "TOTALAREA_MODE": "Home building size",
    "YEARS_BEGINEXPLUATATION_AVG": "Home building age (normalised)", "YEARS_BUILD_AVG": "Year home building was built (normalised)",
    "FLAG_OWN_REALTY": "Owns a house or flat", "NAME_TYPE_SUITE": "Who came with the applicant",
    "FLAG_MOBIL": "Gave a mobile number", "FLAG_EMP_PHONE": "Gave an employer phone number", "FLAG_WORK_PHONE": "Gave a work landline",
    "FLAG_CONT_MOBILE": "Mobile number was reachable", "FLAG_PHONE": "Gave a home phone", "FLAG_EMAIL": "Gave an email address",
    "REG_REGION_NOT_LIVE_REGION": "Lives outside registered region", "REG_REGION_NOT_WORK_REGION": "Works outside registered region",
    "LIVE_REGION_NOT_WORK_REGION": "Works in a different region from home", "REG_CITY_NOT_LIVE_CITY": "Lives outside registered city",
    "REG_CITY_NOT_WORK_CITY": "Works outside registered city", "LIVE_CITY_NOT_WORK_CITY": "Works in a different city from home",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Social circle: people observed 30+ days late", "DEF_30_CNT_SOCIAL_CIRCLE": "Social circle: people who defaulted (30 days)",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Social circle: people observed 60+ days late", "DEF_60_CNT_SOCIAL_CIRCLE": "Social circle: people who defaulted (60 days)",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Credit enquiries in the last hour", "AMT_REQ_CREDIT_BUREAU_DAY": "Credit enquiries in the last day",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Credit enquiries in the last week", "AMT_REQ_CREDIT_BUREAU_MON": "Credit enquiries in the last month",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Credit enquiries in the last 3 months", "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit enquiries in the last year",
    "EXT_SOURCE_1_MISSING": "External credit score 1 not available", "EXT_SOURCE_3_MISSING": "External credit score 3 not available",
    "AMT_GOODS_PRICE_MISSING": "Price of goods not given", "N_CONTACTS": "Contact details provided",
    "BUR_N_BAD_DEBT": "Loans marked bad debt at other lenders", "BUR_N_CREDIT_TYPES": "Types of credit at other lenders",
    "BUR_TOTAL_OVERDUE": "Amount overdue at other lenders", "BUR_N_OVERDUE_NOW": "Loans overdue now at other lenders",
    "BUR_N_PROLONGED": "Times other loans were extended", "BB_N_MONTHS": "Months of history at other lenders",
    "BB_N_SEVERE_LATE_MONTHS": "Months 60+ days late at other lenders", "PREV_N_APPS": "Number of past applications",
    "PREV_CANCELED_SHARE": "Share of past applications cancelled", "POS_MAX_DPD": "Most days late on store/cash loans",
    "CC_N_CARDS": "Number of past credit cards", "CC_BELOW_MIN_PAYMENT_SHARE": "Share of months paying below card minimum",
    "CC_MAX_DPD": "Most days late on credit card", "HAS_PREV_APPLICATION": "Applied here before",
}

# Loan terms the applicant can realistically change: the levers for a "path to approval".
ACTIONABLE = {"AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CREDIT_TERM", "CREDIT_TO_INCOME",
              "ANNUITY_TO_INCOME", "CREDIT_TO_GOODS", "NAME_CONTRACT_TYPE"}

# Personal characteristics that must never be suggested as something to change,
# and that a decision letter should not cite as a reason.
SENSITIVE = {"AGE_YEARS", "NAME_FAMILY_STATUS", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "CODE_GENDER", "EMPLOYED_TO_AGE"}


def category_of(feature: str) -> str:
    if feature in ACTIONABLE:
        return "actionable"
    if feature in SENSITIVE:
        return "sensitive"
    if feature.startswith("EXT_SOURCE"):
        return "external score"
    if feature.startswith(("BUR_", "BB_", "PREV_", "INST_", "POS_", "CC_", "HAS_")):
        return "credit history"
    return "profile"


def _dictionary() -> pd.Series:
    path = ROOT / "docs" / "feature_dictionary.csv"
    return pd.read_csv(path).set_index("feature")["description"] if path.exists() else pd.Series(dtype=str)


_DESC = None


def label_of(feature: str) -> str:
    global _DESC
    if feature in LABELS:
        return LABELS[feature]
    if _DESC is None:
        _DESC = _dictionary()
    d = _DESC.get(feature, "")
    if isinstance(d, str) and d.startswith("Normalized information about building"):
        return f"Home building details ({feature.removesuffix('_AVG').replace('_', ' ').lower()})"
    return (d[:60] + "...") if isinstance(d, str) and len(d) > 60 else (d or feature)


# Building features named YEARS_* are normalised scores, not lengths of time
NORMALISED = {"YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG"}


def format_duration(years: float) -> str:
    """A length of time in words: 'about 5 weeks', 'about 7 months', '3.2 years' (never '0.1 years')."""
    years = abs(float(years))
    days = round(years * 365.25)
    if days < 7:
        return "less than a week"
    if days < 8 * 7:                      # up to 8 weeks: "about 5 weeks" reads better than "about 1 month"
        weeks = round(days / 7)
        return f"about {weeks} week{'s' if weeks != 1 else ''}"
    months = round(years * 12)
    if months < 12:
        return f"about {months} month{'s' if months != 1 else ''}"
    return f"{max(years, 1.0):.1f} years"


def format_value(feature: str, value) -> str:
    """Show a raw feature value the way a person would read it."""
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):       # catches None, float NaN and numpy NaN (e.g. float32)
        return "not available"
    if feature.startswith("AMT_") or feature.startswith(("BUR_TOTAL", "INST_TOTAL", "CC_AVG_BALANCE", "CC_AVG_DRAWINGS",
                                                         "PREV_AVG_AMT", "INCOME_PER_PERSON")):
        return f"{value:,.0f}"
    if "SHARE" in feature or feature in ("BUR_DEBT_TO_CREDIT", "CC_AVG_UTILISATION"):
        return f"{value:.0%}"
    if feature.startswith("EXT_SOURCE") and "N_MISSING" not in feature:
        return f"{value:.2f}"
    if feature in ("CREDIT_TO_INCOME", "CREDIT_TO_GOODS", "ANNUITY_TO_INCOME", "PREV_AVG_GRANTED_TO_APPLIED"):
        return f"{value:.2f}x"
    if feature in ("CREDIT_TERM", "PREV_AVG_TERM"):
        return f"about {value:.0f} payments"
    if feature == "INST_MAX_DAYS_LATE":       # payment date minus due date: 0 or below means no payment was ever late
        return "never late" if value <= 0 else f"{value:.0f} day{'s' if round(value) != 1 else ''}"
    if feature == "OWN_CAR_AGE":
        return f"{value:.0f} year{'s' if round(value) != 1 else ''} old"
    if "YEARS" in feature and feature not in NORMALISED:
        return format_duration(value)
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


class Explainer:
    """Wraps a SHAP TreeExplainer for the SecondLook LightGBM model."""

    def __init__(self, model):
        self.model = model
        self.tree = shap.TreeExplainer(model)
        base = self.tree.expected_value
        self.base_value = float(np.ravel(base)[-1])   # log-odds of the average applicant (class 1)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """PD from either the scikit-learn LGBMClassifier or a plain LightGBM Booster (used by the app)."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values in log-odds, one column per feature."""
        sv = self.tree.shap_values(X)
        return np.asarray(sv[1] if isinstance(sv, list) else sv)

    def explain(self, X_row: pd.DataFrame, top_n: int = 5) -> dict:
        """Explain one applicant (a one-row DataFrame).

        Returns the predicted PD and two tables: the factors that raised risk most and those
        that lowered it most, each with a readable label, the applicant's value and a category.
        """
        if len(X_row) != 1:
            raise ValueError("explain() takes exactly one applicant (one row)")
        sv = self.shap_values(X_row)[0]
        raw = X_row.iloc[0]
        table = pd.DataFrame({
            "feature": X_row.columns,
            "label": [label_of(f) for f in X_row.columns],
            "value": [format_value(f, raw[f]) for f in X_row.columns],
            "shap": sv,
            "category": [category_of(f) for f in X_row.columns],
        })
        table["direction"] = np.where(table["shap"] > 0, "raises risk", "lowers risk")
        pd_hat = float(self.predict_pd(X_row)[0])
        return {
            "pd": pd_hat,
            "log_odds": self.base_value + sv.sum(),
            "base_pd": float(1 / (1 + np.exp(-self.base_value))),
            "raises": table[table["shap"] > 0].nlargest(top_n, "shap").reset_index(drop=True),
            "lowers": table[table["shap"] < 0].nsmallest(top_n, "shap").reset_index(drop=True),
            "all": table.reindex(table["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True),
        }
