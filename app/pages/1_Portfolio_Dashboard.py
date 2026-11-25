"""Portfolio dashboard: risk, approvals and profit across the test applicants."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import BRAND, CATEGORICAL, PE, M, approvals_at_rate, footer, load_model, load_personas, load_portfolio, page_setup

page_setup("Portfolio Dashboard")

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

tab_overview, tab_personas = st.tabs(["Overview", "Personas"])

with tab_overview:
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
                           color_discrete_map={"Repaid": BRAND["teal"], "Defaulted": BRAND["red"]})
        fig.add_vrect(x0=cutoff, x1=cutoff + 0.05, fillcolor=BRAND["amber"], opacity=0.18, line_width=0)
        fig.add_vline(x=cutoff, line_dash="dash", line_color=BRAND["navy"], annotation_text=f"cutoff {cutoff:.2f}", annotation_position="top right")
        fig.update_layout(xaxis_title="Predicted probability of default (capped at 0.6)", yaxis_title="Applicants", height=380, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shaded band = under review (just above the cutoff). Most applicants have low predicted risk. Defaulters (red) cluster at higher PD, but plenty of repayers sit above the cutoff too.")
    with c2:
        st.subheader("Profit by cutoff")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["cutoff"], y=curve["profit"] / 1e9, name="SecondLook model", line=dict(color=BRAND["teal"], width=3)))
        fig.add_hline(y=profit_all / 1e9, line_dash="dot", line_color=BRAND["slate_light"], annotation_text="approve everyone", annotation_position="bottom right")
        fig.add_vline(x=cutoff, line_dash="dash", line_color=BRAND["navy"])
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
                     color="Series", barmode="group", color_discrete_map={"Predicted": BRAND["teal"], "Actual": BRAND["navy"]})
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
                     color_discrete_map={"External score only": BRAND["slate_light"], "SecondLook model": BRAND["teal"]})
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

# ---------------------------------------------------------------- personas
with tab_personas:
    pers = load_personas()
    prof = PE.profile(pers).sort_values("Applicants", ascending=False)
    st.markdown(f"**Who SecondLook helps:** the {len(pers):,} test applicants the model does not approve at the shipped cutoff "
                f"({meta['cutoff']:.2f}), grouped by K-Means into five personas from eight non-sensitive features "
                "(no age, gender or family). The cutoff slider does not change this tab.")
    p1, p2, p3 = st.columns(3)
    p1.metric("Not approved", f"{len(pers):,}")
    p2.metric("Would have repaid", f"{1 - pers['TARGET'].mean():.0%}")
    p3.metric("Have a path to approval", f"{pers['has_path'].mean():.0%}", help="A smaller loan or more payments that the model would approve (Step 8)")
    m1, m2 = st.columns([3, 2])
    with m1:
        fig = px.scatter(pers.sample(frac=1, random_state=42), x="PC1", y="PC2", color="persona", category_orders={"persona": list(prof.index)}, color_discrete_sequence=CATEGORICAL,
                         opacity=0.55, render_mode="webgl", hover_data={"PC1": False, "PC2": False, "PD": ":.1%"})
        fig.update_traces(marker_size=4)
        lo, hi = pers[["PC1", "PC2"]].quantile(0.005), pers[["PC1", "PC2"]].quantile(0.995)
        fig.update_layout(height=470, margin=dict(t=30), legend=dict(title_text="", orientation="h", y=-0.22), xaxis_range=[lo["PC1"], hi["PC1"]], yaxis_range=[lo["PC2"], hi["PC2"]],
                          xaxis_title="PC1: loan size for the income (25% of variance)", yaxis_title="PC2: credit history (20%)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("PCA map of the eight scaled features: the two directions shown keep 45% of the variance, so groups that "
                   "differ mainly in past late payments overlap here.")
    with m2:
        st.markdown("**The five personas**")
        for name, r in prof.iterrows():
            st.markdown(f"**{name}** ({int(r['Applicants']):,}, {r['Share']:.0%}): {PE.describe(name, r)}")
    st.markdown("**Persona profiles** (medians unless stated)")
    show = prof.rename(columns=PE.FEATURE_LABELS)
    pct = ["Share", "Payment ÷ income", "Share of past payments late", "Has a bureau record", "Repayment rate", "Mean PD", "Path to approval"]
    st.dataframe(show.style.format({**{c: "{:.0%}" for c in pct}, "Applicants": "{:,}", "External credit score": "{:.3f}",
                                    "Loan ÷ yearly income": "{:.1f}x", "Number of payments": "{:.0f}", "Years in current job": "{:.1f}",
                                    "Loans at other lenders": "{:.0f}"}), use_container_width=True)
    st.caption("About seven in ten of every persona actually repaid: the personas differ in why they were not approved, not in how "
               "risky they are, so each needs a different response. *Has a bureau record* is a share. Method: notebooks/09_personas.ipynb.")

footer()
