"""SecondLook: home page."""
import streamlit as st

from lib import STATIC, footer, load_model, load_portfolio, page_setup

page_setup()
booster, meta, _ = load_model()
portfolio = load_portfolio()
m = meta["metrics"]

st.image(str(STATIC / "logo_full.png"), width=260)
st.markdown("#### An AI-assisted second review for borderline loan declines")

st.markdown(
    """
About half of Filipino adults have no formal financial account, and **thin credit files** get many of them declined,
usually with no explanation. Lenders decline applicants they can't read; most of those applicants would have repaid.

**Business question:** *Among applicants a lender would decline, which ones are actually safe to approve,
and what realistic changes would get the rest approved?*
"""
)

c1, c2, c3, c4 = st.columns(4)
declined = portfolio["DECISION"].eq("decline")
c1.metric("Model accuracy (ROC AUC)", f"{m['test']['roc_auc']:.3f}", help="On 61,503 test applicants the model never saw. 0.5 = random, 1.0 = perfect.")
c2.metric("Approval rate at the cutoff", f"{m['approval_rate']:.0%}", help=f"Approve if predicted risk (PD) < {meta['cutoff']:.2f}")
c3.metric("Default rate of approved loans", f"{m['default_rate_approved']:.1%}",
          delta=f"{(m['default_rate_approved'] - portfolio['TARGET'].mean()) * 100:+.1f} pts vs approving all", delta_color="inverse")
c4.metric("Not approved, but would repay", f"{1 - portfolio.loc[declined, 'TARGET'].mean():.0%}",
          help="Share of test applicants above the cutoff who actually repaid: SecondLook's opportunity.")

st.subheader("How SecondLook works")
STEPS = [
    ("Score", "A LightGBM model trained on 246,000 past loans estimates each applicant's probability of default (PD), using the application **and** credit history (SQL features from 6 history tables)."),
    ("Flag", "Applicants just above the cutoff are flagged **under review** instead of auto-declined: the borderline group where a second look matters most."),
    ("Explain", "SHAP shows which factors raised or lowered each applicant's risk, in plain English. Sensitive traits are labelled and never used as reasons."),
    ("Offer and communicate", "A re-scored *path to approval* finds the smallest change to the loan (borrow less, or more payments) that would be approved. "
     "Claude drafts a clear, respectful letter from the reasons and that offer. Rule checks and an independent audit "
     "(claims vs the model's reasons, and tone) flag anything for the loan officer to review before sending."),
]
for i, (col, (name, text)) in enumerate(zip(st.columns(4), STEPS), 1):
    with col.container(border=True, height="stretch"):
        st.markdown(f"<span class='sl-step'>STEP {i}</span>", unsafe_allow_html=True)
        st.markdown(f"**{name}**  \n{text}")

st.subheader("Explore the app")
p1, p2, p3, p4 = st.columns(4)
with p1:
    st.page_link("pages/1_Portfolio_Dashboard.py", label="Portfolio Dashboard", icon=":material/monitoring:",
                 help="Risk, approvals and profit across all test applicants; try other cutoffs.")
    st.caption("How the model performs across the whole portfolio, with a live cutoff and profit simulator.")
with p2:
    st.page_link("pages/2_Applicant_Review.py", label="Applicant Review", icon=":material/person_search:",
                 help="Open an applicant: score, reasons and letter.")
    st.caption("What a loan officer sees for one applicant: score, decision, SHAP reasons, path to approval and the drafted letter.")
with p3:
    st.page_link("pages/3_Model_Card.py", label="Model Card", icon=":material/description:",
                 help="How the model was built, its limits and fairness checks.")
    st.caption("How the model was trained and evaluated, its limitations, and fairness checks.")
with p4:
    st.page_link("pages/4_AI_Analyst.py", label="AI Analyst", icon=":material/forum:", help="Ask portfolio questions in plain English.")
    st.caption("Ask a question in plain English; Claude answers with SQL and SecondLook's own tools, and shows its work.")

st.info("**New in Phase 2:** the re-scored *path to approval*, an independent faithfulness and tone audit of every letter, "
        "a full fairness audit on the Model Card, borrower personas on the dashboard, and an AI Analyst that answers "
        "portfolio questions with SQL.", icon=":material/new_releases:")
footer()
