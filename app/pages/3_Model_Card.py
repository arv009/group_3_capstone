"""Model card: how the model was built, how well it works, its limits and fairness checks."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Card · SecondLook", layout="wide")

from lib import applicant_X, footer, label_of, load_applicants, load_fairness, load_model, load_portfolio  # noqa: E402

booster, meta, explainer = load_model()
df = load_portfolio()
m = meta["metrics"]

st.title("Model Card")
st.caption("A model card summarises what a model is for, how it was built and tested, and where it should not be trusted.")

st.subheader("Model details")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
| | |
|---|---|
| **Task** | Predict the probability that a loan applicant will have payment difficulties (default) |
| **Algorithm** | LightGBM gradient-boosted trees, {meta['n_trees']:,} trees, learning rate {meta['params']['learning_rate']} |
| **Inputs** | {len(meta['features'])} features: application data, engineered ratios, and SQL summaries of 6 credit-history tables |
| **Excluded on purpose** | Gender (kept only for the fairness audit), applicant ID |
| **Training data** | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data): 246,008 applicants (80%) |
| **Test data** | 61,503 applicants (20%), stratified, never used in training |
| **Decision rule** | Approve if PD < {meta['cutoff']:.2f}; PD {meta['borderline_band'][0]:.2f}–{meta['borderline_band'][1]:.2f} flagged *under review* |
""")
with c2:
    st.markdown(f"""
| Metric (test set) | LightGBM | Logistic baseline |
|---|---|---|
| ROC AUC | **{m['test']['roc_auc']:.3f}** | {m['baseline_test']['roc_auc']:.3f} |
| PR AUC | **{m['test']['pr_auc']:.3f}** | {m['baseline_test']['pr_auc']:.3f} |
| Brier score (lower is better) | **{m['test']['brier']:.4f}** | {m['baseline_test']['brier']:.4f} |
| 5-fold CV AUC | {m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f} | |
| Precision / recall of declines at cutoff | {m['precision_at_cutoff']:.0%} / {m['recall_at_cutoff']:.0%} | |
| Approval rate / default rate of approved | {m['approval_rate']:.0%} / {m['default_rate_approved']:.1%} | |
""")
    pa = meta["profit_assumptions"]
    st.caption(f"Cutoff chosen to maximise profit under stated assumptions: +{pa['margin']:.0%} of the loan on repayment, "
               f"−{pa['lgd']:.0%} on default.")

st.subheader("What drives the model (SHAP)")


@st.cache_data(show_spinner="Computing SHAP importance...")
def shap_importance():
    ids = load_applicants()["SK_ID_CURR"].tolist()
    X = pd.concat([applicant_X(i) for i in ids])
    sv = explainer.shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False).head(15)
    return pd.DataFrame({"Factor": [label_of(f) for f in imp.index], "Mean |SHAP|": imp.values})


imp = shap_importance()
fig = px.bar(imp.iloc[::-1], x="Mean |SHAP|", y="Factor", orientation="h", color_discrete_sequence=["#55A868"])
fig.update_layout(height=450, margin=dict(t=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("Average size of each factor's push on the score, over the 300 applicants in this demo. "
           "External credit scores dominate, followed by past repayment behaviour and loan structure.")

st.subheader("Fairness check (preview)")
st.markdown("Gender is **not** a model input. We still check whether outcomes differ between groups, because other features can act as proxies.")
approved = df["PD_LGBM"] < meta["cutoff"]
df = df.assign(Approved=approved, Age_group=pd.cut(df["AGE_YEARS"], [20, 30, 40, 50, 60, 70], labels=["20-29", "30-39", "40-49", "50-59", "60-69"]))


def fairness(by):
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({
        "Applicants": g.size(),
        "Default rate": g["TARGET"].mean(),
        "Approval rate": g["Approved"].mean(),
        "Approval rate of repayers": df[df["TARGET"] == 0].groupby(by, observed=True)["Approved"].mean(),
        "Default rate of approved": df[df["Approved"]].groupby(by, observed=True)["TARGET"].mean(),
    })
    return t[t["Applicants"] >= 100]          # tiny groups (e.g. 2 'unknown' gender) give unreliable rates


fmt = {"Applicants": "{:,}", "Default rate": "{:.1%}", "Approval rate": "{:.1%}", "Approval rate of repayers": "{:.1%}",
       "Default rate of approved": "{:.1%}"}
fg = fairness(df["CODE_GENDER"].rename("Gender"))
fa = fairness(df["Age_group"].rename("Age group"))
st.markdown("**By gender**")
st.dataframe(fg.style.format(fmt), use_container_width=True)
st.markdown("**By age group**")
st.dataframe(fa.style.format(fmt), use_container_width=True)
st.markdown(f"""
**What this shows**
- Approval rates follow each group's actual default rate, and in **every** group the approved loans default less often than the group overall.
- **Equal-opportunity gap:** among applicants who actually **repaid**, approval rates still differ:
  **{fg.loc['M', 'Approval rate of repayers']:.0%}** for men vs **{fg.loc['F', 'Approval rate of repayers']:.0%}** for women, and
  **{fa['Approval rate of repayers'].iloc[0]:.0%}** for ages 20-29 vs **{fa['Approval rate of repayers'].iloc[-1]:.0%}** for ages 60-69.
  Good young and male borrowers are declined more often than good older and female ones.
- Gender is not a model input, but age is, and other features (job, employer, external scores) correlate with both. This is the kind of
  gap a **human second look** is meant to catch.
""")

# ---------------------------------------------------------------- Phase 2 fairness audit (notebook 08)
fair = load_fairness()
st.subheader("Fairness audit: how sure are we, and does removing age help?")
gaps = pd.DataFrame(fair["gaps"])
eo = gaps[gaps["metric"] == "repayer_approval"]
st.dataframe(pd.DataFrame({
    "Comparison": eo["a"] + " vs " + eo["b"],
    "Gap in approval rate of repayers": [f"{g * 100:+.1f} pts" for g in eo["gap"]],
    "95% interval (1,000 bootstrap resamples)": [f"{lo * 100:.1f} to {hi * 100:.1f}" for lo, hi in zip(eo["low"], eo["high"])],
    "Reading": ["real (interval excludes zero)" if r else "could be noise" for r in eo["real"]],
}), hide_index=True, use_container_width=True)
exp = pd.DataFrame(fair["experiment"]).set_index("metric")
exp.columns = ["Shipped model", "Retrained without age and family features"]
st.dataframe(exp.loc[["AUC", "profit at 0.17 (M)", "EO gap women vs men", "EO gap 60s vs 20s",
                      "EO gap bureau history vs thin file"]].rename(index={
                 "profit at 0.17 (M)": "Profit at the 0.17 cutoff (millions)",
                 "EO gap women vs men": "Gap: women vs men (pts, 95% CI)", "EO gap 60s vs 20s": "Gap: 60s vs 20s (pts, 95% CI)",
                 "EO gap bureau history vs thin file": "Gap: bureau history vs thin file (pts, 95% CI)"}),
             use_container_width=True)
st.markdown("""
- **Every gap is real**, and no group falls under the four-fifths rule of thumb (lowest approval ratio 0.82, people in their 20s).
  The model is calibrated in every group, so the gaps reflect real differences in default rates meeting one cutoff.
- **Removing age, employment-to-age, family status, children and family size** lowered AUC slightly and **widened** the age gap
  (good borrowers in their 20s approved 81.3% instead of 83.1%): age is still visible through job tenure and credit history.
- **Decision:** we keep the shipped model. Sensitive features are never cited in letters or suggested as changes, and the
  borderline band gets a loan officer's second look.
""")

st.subheader("Intended use and limitations")
st.markdown("""
- **Intended use:** decision *support* for loan officers reviewing consumer loan applications, especially borderline ones. A human makes the final decision.
- **Not for:** fully automated declines, or markets and products very different from the training data.
- **Data:** historical, anonymised Home Credit data; the competition does not disclose which country it comes from. Before real use, the model would need to be validated on local (e.g. Philippine) applicants.
- **Probabilities** are well calibrated on the test set, but calibration should be monitored over time as the applicant mix changes.
- **Explanations** (SHAP) describe *the model's* reasoning, not causes in the real world.
- **Letters** are drafted by Claude and checked by rules; rules can't verify meaning, so a loan officer reviews every letter. Phase 2 adds an independent faithfulness check.
- **Profit figures** use illustrative assumptions, adjustable on the dashboard.
""")
footer()
