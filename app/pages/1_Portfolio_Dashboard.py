"""Portfolio dashboard: risk, approvals and profit across the test applicants."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Portfolio Dashboard · SecondLook", layout="wide")

from lib import M, approvals_at_rate, footer, load_model, load_personas, load_portfolio  # noqa: E402

_, meta, _ = load_model()
df = load_portfolio()
y, pd_hat, amount = df["TARGET"].values, df["PD_LGBM"].values, df["AMT_CREDIT"].values

st.title("Portfolio Dashboard")
st.caption(f"{len(df):,} held-out test applicants the model never saw during training. "
           "Move the cutoff to see how approvals, defaults and profit change.")

# ---------------------------------------------------------------- controls
with st.sidebar:
    st.header("Decision settings")
    cutoff = st.slider("Approve if predicted risk (PD) is below", 0.05, 0.40, float(meta["cutoff"]), 0.01,
                       help=f"The model's profit-maximising cutoff is {meta['cutoff']:.2f}.")
    st.subheader("Profit assumptions")
    margin = st.slider("Profit on a repaid loan (% of loan)", 5, 30, int(meta["profit_assumptions"]["margin"] * 100), 1) / 100
    lgd = st.slider("Loss on a defaulted loan (% of loan)", 30, 90, int(meta["profit_assumptions"]["lgd"] * 100), 5) / 100
    st.caption("Illustrative unit economics, not Home Credit's real figures.")

assumptions = M.ProfitAssumptions(margin=margin, lgd=lgd)
approved = pd_hat < cutoff
band = (pd_hat >= cutoff) & (pd_hat < cutoff + 0.05)
declined = pd_hat >= cutoff + 0.05
curve = M.profit_curve(y, pd_hat, amount, assumptions)
profit_now = M.profit_curve(y, pd_hat, amount, assumptions, cutoffs=[cutoff])["profit"].iloc[0]
profit_all = M.approve_all_profit(y, amount, assumptions)
best = curve.loc[curve["profit"].idxmax()]

tab_portfolio, tab_personas = st.tabs(["Portfolio", "Personas"])

with tab_portfolio:
    # ---------------------------------------------------------------- KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Approved", f"{approved.mean():.1%}", help="PD below the cutoff")
    k2.metric("Under review", f"{band.mean():.1%}", help="PD within 5 points above the cutoff: SecondLook's borderline group")
    k3.metric("Default rate of approved", f"{y[approved].mean():.2%}", delta=f"{(y[approved].mean() - y.mean()) * 100:+.2f} pts vs approve all", delta_color="inverse")
    k4.metric("Profit vs approving everyone", f"{profit_now / profit_all - 1:+.1%}")
    k5.metric("Not approved, would repay", f"{1 - y[~approved].mean():.0%}", help="Share of applicants above the cutoff who actually repaid")

    if abs(cutoff - best["cutoff"]) > 0.005:
        st.caption(f"With these assumptions, the most profitable cutoff is **{best['cutoff']:.2f}** "
                   f"({best['profit'] / profit_all - 1:+.1%} vs approving everyone).")

    # ---------------------------------------------------------------- charts row 1
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Where applicants fall")
        plot = pd.DataFrame({"PD": np.clip(pd_hat, 0, 0.6), "Outcome": np.where(y == 1, "Defaulted", "Repaid")})
        fig = px.histogram(plot, x="PD", color="Outcome", nbins=60, barmode="stack",
                           color_discrete_map={"Repaid": "#4C72B0", "Defaulted": "#C44E52"})
        fig.add_vrect(x0=cutoff, x1=cutoff + 0.05, fillcolor="#E08E0B", opacity=0.18, line_width=0)
        fig.add_vline(x=cutoff, line_dash="dash", line_color="black", annotation_text=f"cutoff {cutoff:.2f}", annotation_position="top right")
        fig.update_layout(xaxis_title="Predicted probability of default (capped at 0.6)", yaxis_title="Applicants", height=380, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shaded band = under review (just above the cutoff). Most applicants have low predicted risk. Defaulters (red) cluster at higher PD, but plenty of repayers sit above the cutoff too.")
    with c2:
        st.subheader("Profit by cutoff")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["cutoff"], y=curve["profit"] / 1e9, name="SecondLook model", line=dict(color="#C44E52", width=3)))
        fig.add_hline(y=profit_all / 1e9, line_dash="dot", line_color="grey", annotation_text="approve everyone", annotation_position="bottom right")
        fig.add_vline(x=cutoff, line_dash="dash", line_color="black")
        fig.update_layout(xaxis_title="PD cutoff", yaxis_title="Profit on test set (billions)", height=380, margin=dict(t=30), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Profit peaks where the value of extra good loans is balanced by the cost of extra defaults.")

    # ---------------------------------------------------------------- charts row 2
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Predicted vs actual risk")
        dec = pd.qcut(pd_hat, 10, labels=[f"D{i}" for i in range(1, 11)])
        calib = pd.DataFrame({"Predicted": pd_hat, "Actual": y}).groupby(dec, observed=True).mean().mul(100).reset_index(names="Decile")
        fig = px.bar(calib.melt(id_vars="Decile", var_name="Series", value_name="Default rate (%)"), x="Decile", y="Default rate (%)",
                     color="Series", barmode="group", color_discrete_map={"Predicted": "#DD8452", "Actual": "#4C72B0"})
        fig.update_layout(height=360, margin=dict(t=30), xaxis_title="Risk decile (D1 = safest 10%)", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Predicted and actual default rates match in every decile: the probabilities are honest, not just well-ranked.")
    with c4:
        st.subheader("More approvals at the same risk")
        ext = -pd.Series(df["EXT_SOURCE_MEAN"]).fillna(df["EXT_SOURCE_MEAN"].median()).values
        rows = []
        for r in (0.04, 0.05, 0.06):
            rows += [{"Default rate allowed": f"{r:.0%}", "Policy": "External score only", "Approvals": approvals_at_rate(y, ext, r)},
                     {"Default rate allowed": f"{r:.0%}", "Policy": "SecondLook model", "Approvals": approvals_at_rate(y, pd_hat, r)}]
        gain = pd.DataFrame(rows)
        fig = px.bar(gain, x="Default rate allowed", y="Approvals", color="Policy", barmode="group",
                     color_discrete_map={"External score only": "#8C8C8C", "SecondLook model": "#C44E52"})
        fig.update_layout(height=360, margin=dict(t=30), legend_title_text="")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)
        g5 = gain.pivot(index="Default rate allowed", columns="Policy", values="Approvals").loc["5%"]
        st.caption(f"At a 5% default rate the model approves **{g5['SecondLook model'] - g5['External score only']:,} more applicants "
                   f"(+{g5['SecondLook model'] / g5['External score only'] - 1:.0%})** than ranking by external score alone.")

    # ---------------------------------------------------------------- thin file
    st.subheader("Thin-file applicants (no credit bureau record)")
    thin = df["HAS_BUREAU_HISTORY"].eq(0).values
    t = pd.DataFrame({
        "Group": ["No bureau record (thin file)", "Has bureau history"],
        "Applicants": [int(thin.sum()), int((~thin).sum())],
        "Actual default rate": [y[thin].mean(), y[~thin].mean()],
        "Approved at this cutoff": [approved[thin].mean(), approved[~thin].mean()],
        "Default rate of approved": [y[thin & approved].mean(), y[~thin & approved].mean()],
    })
    st.dataframe(t.style.format({"Applicants": "{:,}", "Actual default rate": "{:.1%}", "Approved at this cutoff": "{:.1%}", "Default rate of approved": "{:.1%}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Thin-file applicants are riskier as a group, but the model still approves most of them, at a much lower default rate than the group average.")

with tab_personas:
    pers, prof = load_personas()
    st.markdown(f"**Who does the model turn away?** K-Means clustering groups the {len(pers):,} test applicants above the "
                f"{meta['cutoff']:.2f} cutoff into {len(prof)} types, using 8 loan and credit-record features "
                "(never age, gender or family). Notebook 09 explains the choices.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Not approved", f"{len(pers):,}")
    m2.metric("Would have repaid", f"{1 - pers['TARGET'].mean():.0%}")
    m3.metric("Have a path to approval", f"{pers['has_path'].mean():.0%}", help="A smaller loan or longer term (notebook 06)")
    left, right = st.columns([3, 2])
    with left:
        sample = pers.sample(min(3000, len(pers)), random_state=1)
        fig = px.scatter(sample, x="pc1", y="pc2", color="persona", opacity=0.6, height=430,
                         labels={"pc1": "Principal component 1", "pc2": "Principal component 2", "persona": "Persona"})
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(margin=dict(t=30), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Each dot is an applicant, placed by PCA on the two directions with the most variation "
                   f"({prof['pca_variance_explained'].iloc[0]:.0%} of it). The types overlap: they are helpful labels, not sharp segments.")
    with right:
        for r in prof.itertuples():
            st.markdown(f"**{r.persona}** · {r.applicants:,} applicants  \n{r.sentence}  \n"
                        f"Repaid {r.repaid:.0%} · path to approval {r.has_path:.0%}")
    table = prof.set_index("persona")[["applicants", "repaid", "has_path", "under_review", "EXT_SOURCE_MEAN",
                                       "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TERM", "YEARS_EMPLOYED",
                                       "INST_LATE_SHARE", "HAS_BUREAU_HISTORY", "BUR_N_LOANS"]]
    table.columns = ["Applicants", "Repaid", "Path to approval", "Under review", "External score", "Loan / income",
                     "Repayment / income", "Payments", "Years employed", "Late payments", "Bureau record", "Past loans"]
    st.dataframe(table.style.format({"Applicants": "{:,}", "Repaid": "{:.0%}", "Path to approval": "{:.0%}",
                                     "Under review": "{:.0%}", "External score": "{:.2f}", "Loan / income": "{:.1f}",
                                     "Repayment / income": "{:.2f}", "Payments": "{:.0f}", "Years employed": "{:.1f}",
                                     "Late payments": "{:.0%}", "Bureau record": "{:.0f}", "Past loans": "{:.0f}"}),
                 use_container_width=True)
    st.caption("Medians per persona. Every type repays about 71-72% of the time: there are good borrowers to find in each group.")

footer()
