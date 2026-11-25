"""Model card: how the model was built, how well it works, its limits and fairness checks."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from lib import (BRAND, applicant_X, footer, label_of, load_applicants, load_audits, load_fairness, load_model, load_personas,
                 page_setup)

page_setup("Model Card")

booster, meta, explainer = load_model()
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
fig = px.bar(imp.iloc[::-1], x="Mean |SHAP|", y="Factor", orientation="h", color_discrete_sequence=[BRAND["teal"]])
fig.update_layout(height=450, margin=dict(t=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("Average size of each factor's push on the score, over the 300 applicants in this demo. "
           "External credit scores dominate, followed by past repayment behaviour and loan structure.")

st.subheader("What SecondLook adds to the model (Phase 2)")
pers = load_personas()
has_path = pers.groupby("DECISION")["has_path"].mean()
audits = load_audits().values()
n_claims, n_ok = sum(a["n_claims"] for a in audits), sum(a["n_supported"] for a in audits)
repaid = 1 - pers.groupby("persona", observed=True)["TARGET"].mean()
largest = pers["persona"].value_counts(normalize=True)
st.markdown(f"""
| Part | Result | How it was checked |
|---|---|---|
| **Path to approval** | **{has_path['under_review']:.1%}** of under-review and **{has_path['decline']:.1%}** of declined applicants have one (borrow less, or more payments) | Every option is re-scored by the real model and must land at least 0.5 points under the cutoff; only loan terms change (notebook 06) |
| **Letter audit** | **{n_ok / n_claims:.1%}** of the {n_claims} claims in the {len(audits)} demo letters are supported by the model's reasons; {sum(not a['issues'] for a in audits)} letters have no flag | A second Claude lists the claims, Python checks them against the SHAP facts; it caught all 4 planted errors, but only 9 of its 16 flags were real problems (notebook 07) |
| **Borrower personas** | {len(pers):,} not-approved applicants in 5 K-Means groups; the largest is *{largest.index[0]}* ({largest.iloc[0]:.0%}). Repayment is {repaid.min():.0%}-{repaid.max():.0%} in every persona | Silhouette score and an elbow plot to choose k; the groups follow the credit file far more than age or gender (notebook 09) |
| **AI Analyst** | 20 of 20 test answers correct over two runs, about $0.0008 a question | Each expected answer computed independently in pandas; answers read by hand, and off-topic probes tested (notebook 10) |
""")

st.subheader("Fairness audit")
fair = load_fairness()
exp = fair["experiment"]
st.markdown(f"""
Are good borrowers treated equally across **gender**, **age** and **thin credit files**? Measured on all
{fair['n_test']:,} test applicants at the shipped cutoff (approve if PD < {fair['cutoff']:.2f}). Gender is **not** a model
input, but other features can act as proxies; age **is** an input. Full method: `notebooks/08_fairness_audit.ipynb`.
""")

COLS = {"Applicants": "{:,}", "Default rate": "{:.1%}", "Approval rate": "{:.1%}", "Approval rate of repayers": "{:.1%}",
        "Default rate of approved": "{:.1%}", "Mean predicted PD": "{:.1%}", "Four-fifths ratio": "{:.2f}"}
gaps = pd.DataFrame(fair["gaps"])
for tab, (grouping, rows) in zip(st.tabs(list(fair["tables"])), fair["tables"].items()):
    with tab:
        t = pd.DataFrame(rows).set_index("Group")[list(COLS)]
        st.dataframe(t.style.format(COLS), use_container_width=True)
        g = gaps[(gaps["Grouping"] == grouping) & (gaps["Metric"] == "Approval rate of repayers")].iloc[0]
        r = gaps[(gaps["Grouping"] == grouping) & (gaps["Metric"] == "Four-fifths ratio")].iloc[0]
        st.caption(f"**{g['Comparison']}:** repayers approved {g['Gap'] * 100:+.1f} points more often "
                   f"(95% interval {g['95% CI low'] * 100:+.1f} to {g['95% CI high'] * 100:+.1f}, {fair['n_boot']:,} bootstrap resamples). "
                   f"Four-fifths ratio {r['Gap']:.2f} ({r['95% CI low']:.2f} to {r['95% CI high']:.2f}).")

age = {row["Group"]: row for row in fair["tables"]["Age band"]}
st.markdown(f"""
**What this shows**
- **Calibrated in every group:** mean predicted PD is within 1 point of each group's real default rate, so one cutoff means the same risk for everyone.
- **Passes the four-fifths check:** no group's approval rate is below 0.8 of the best group's. Ages 20-29 come closest at **{age['20-29']['Four-fifths ratio']:.2f}**.
- **But good borrowers are not approved equally, and the gaps are real:** {age['20-29']['Approval rate of repayers']:.0%} of repayers aged 20-29 were approved
  vs {age['60-69']['Approval rate of repayers']:.0%} aged 60-69. Men and thin-file repayers are also approved less often. Every bootstrap interval excludes zero.
- Approved loans in those groups still default more, so the model is not harsher on them than their risk warrants. When real default rates differ,
  a model can't be calibrated **and** give repayers equal approval rates at once.
""")

st.markdown("**Fix experiment: retrain without the sensitive features**")
st.caption("Removed: " + ", ".join(f"`{f}`" for f in exp["dropped"]) + ". Same split, settings and tree-count rule.")
ROWS = {"ROC AUC": "{:.4f}", "Approval rate": "{:.1%}", "Good borrowers declined": "{:,}", "Profit (billions)": "{:.3f}",
        "Age band: Approval rate of repayers gap": "pts", "Age band: Four-fifths ratio": "{:.3f}",
        "Gender: Approval rate of repayers gap": "pts", "Credit file: Approval rate of repayers gap": "pts"}


def fmt_cell(row, m):
    v = m[row]
    if ROWS[row] == "pts":
        lo, hi = m[row.replace(" gap", " CI")]
        return f"{v * 100:+.1f} pts ({lo * 100:+.1f} to {hi * 100:+.1f})"
    if "ratio" in row:
        lo, hi = m[row + " CI"]
        return f"{v:.3f} ({lo:.3f} to {hi:.3f})"
    return ROWS[row].format(v)


cmp = pd.DataFrame({name: {r.replace(": Approval rate of repayers gap", ": repayer approval gap"): fmt_cell(r, m) for r in ROWS}
                    for name, m in exp["models"].items()})
st.dataframe(cmp, use_container_width=True)
a = {row["Group"]: row for row in exp["age"]}
st.markdown(f"""
- **Removing age made the age gap worse.** Without age, the model still sees features that go with being young (short job tenure, a new ID,
  a short credit history) but can't judge them in context, so it overstates young applicants' risk
  ({a['20-29']['Mean PD, without sensitive']:.1%} predicted vs {a['20-29']['Actual default rate']:.1%} actual) and the 20-29 band drops **below** the four-fifths line.
- Accuracy dips slightly, profit is a wash (under 0.4% either way), and the gender and thin-file gaps barely move.
- A looser cutoff narrows every gap, but for both models alike; the shipped model at the same cutoff is still fairer by age.
""")
st.success("**Decision: we ship the model with all features.** Removing sensitive features cost accuracy and made the age gap worse; "
           "instead, sensitive traits are never cited as reasons, borderline applicants (where young and thin-file applicants are "
           "over-represented) get a human second look, and declined applicants get a path to approval.")

st.subheader("Intended use and limitations")
st.markdown("""
- **Intended use:** decision *support* for loan officers reviewing consumer loan applications, especially borderline ones. A human makes the final decision.
- **Not for:** fully automated declines, or markets and products very different from the training data.
- **Data:** historical, anonymised Home Credit data; the competition does not disclose which country it comes from. Before real use, the model would need to be validated on local (e.g. Philippine) applicants.
- **Probabilities** are well calibrated on the test set, but calibration should be monitored over time as the applicant mix changes.
- **Explanations** (SHAP) describe *the model's* reasoning, not causes in the real world.
- **Letters** are drafted by Claude and checked by rules, then by an independent audit (a second Claude reads only the letter; Python checks its claims against the model's reasons). In testing it caught all four planted errors, but about half of its flags on real letters were false alarms and its verdicts vary between runs, so a loan officer reviews every letter.
- **Fairness:** good young, male and thin-file borrowers are approved less often than others (see the audit above). Outcomes are only known for past *approved* loans, and intersections (e.g. young men with thin files) were not audited. Re-run the audit on new applicants.
- **Paths to approval** change only the loan amount and number of payments, and are judged by the model alone. They have not been tested with real borrowers, and a smaller loan may not meet the applicant's need.
- **Personas** describe the test applicants the model doesn't approve. They are a summary for planning, not a rule for any one applicant.
- **The AI Analyst** was tested on ten questions we wrote about data we know. Vaguer questions will be less reliable, so every answer shows its SQL and tool calls.
- **Profit figures** use illustrative assumptions, adjustable on the dashboard.
""")
footer()
