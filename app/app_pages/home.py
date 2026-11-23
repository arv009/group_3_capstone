"""Overview page: the problem and the portfolio at a glance."""

import streamlit as st

from secondlook.features import load_features

st.title("SecondLook")
st.caption("Fair, explainable credit scoring for applicants with thin credit files")


@st.cache_data
def portfolio_stats() -> dict:
    df = load_features(with_split=False)
    thin = df["THIN_FILE"] == 1
    return {
        "applicants": len(df),
        "default_rate": df["TARGET"].mean(),
        "thin_share": thin.mean(),
        "thin_ext3_missing": df.loc[thin, "EXT_SOURCE_3"].isna().mean(),
    }


try:
    s = portfolio_stats()
except FileNotFoundError as e:
    st.error(str(e), title="Feature table not found")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Applicants", f"{s['applicants']:,}", border=True)
c2.metric("Default rate", f"{s['default_rate']:.1%}", border=True)
c3.metric("Thin-file applicants", f"{s['thin_share']:.1%}", border=True,
          help="No credit history with any other lender (no credit bureau record).")
c4.metric("Thin-file missing bureau score", f"{s['thin_ext3_missing']:.0%}", border=True,
          help="Share of thin-file applicants with no EXT_SOURCE_3, one of the strongest standard risk scores.")

st.markdown(
    """
**The problem.** One in seven applicants has no credit bureau history. Standard scores lose
their strongest signal for these people, so many who would repay get declined with no explanation.

**What SecondLook does**
1. Scores default risk using alternative data (repayment history, previous applications, card and POS behaviour).
2. Refers uncertain cases to an underwriter instead of guessing.
3. Explains every decision and suggests a realistic path to approval.
4. Writes a plain-language decision letter with Claude, checked against the model.
5. Audits decisions for fairness across gender and age.
"""
)
st.page_link("app_pages/applicant_review.py", label="Review an applicant", icon=":material/person_search:")
