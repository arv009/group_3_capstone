"""Applicant review: decision, reasons, path to approval and decision letter for one held-out applicant."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from secondlook.config import ROOT
from secondlook.explain import describe
from secondlook.letters import ESCALATION_MODEL, MODEL, Usage, build_facts, generate
from secondlook.review import DECISION_TEXT, load_context, review

DECISION_STYLE = {
    "approve": ("green", ":material/check_circle:"),
    "refer": ("orange", ":material/support_agent:"),
    "decline": ("red", ":material/cancel:"),
}
GROUPS = {"Declined": "decline", "Referred": "refer", "Approved": "approve"}
MAX_LISTED = 300  # applicants offered per group (the test split has 46,126)

st.title("Applicant review")
st.caption("Applicants from the held-out test split: the model never saw them or their outcomes.")


@st.cache_resource(show_spinner="Loading the model and scoring applicants...")
def get_context():
    return load_context()


@st.cache_resource
def get_claude_client():
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return anthropic.Anthropic()


try:
    ctx = get_context()
except FileNotFoundError as e:
    st.error(str(e), title="Model or data not found")
    st.stop()

st.session_state.setdefault("letters", {})
A = ctx.applicants

# ------------------------------------------------------------------ pick an applicant
with st.container(horizontal=True, vertical_alignment="bottom"):
    group = st.segmented_control("Decision", list(GROUPS), default="Declined", key="group")
    thin_only = st.toggle("Thin-file only", key="thin_only")

pool = A[A["decision"] == GROUPS[group or "Declined"]]
if thin_only:
    pool = pool[pool["THIN_FILE"] == 1]
ids = sorted(pool.index[:MAX_LISTED].tolist())


if st.session_state.get("applicant") not in ids:  # filters changed: start from the first applicant listed
    st.session_state["applicant"] = ids[0] if ids else None


def pick_random() -> None:
    st.session_state["applicant"] = int(np.random.default_rng().choice(ids))


with st.container(horizontal=True, vertical_alignment="bottom"):
    applicant_id = st.selectbox("Applicant ID", ids, key="applicant",
                                help=f"{len(pool):,} applicants in this group; the first {min(MAX_LISTED, len(pool))} are listed.")
    st.button("Random applicant", icon=":material/shuffle:", on_click=pick_random)

if applicant_id is None:
    st.info("No applicants match these filters.")
    st.stop()

r = review(ctx, int(applicant_id))
color, icon = DECISION_STYLE[r["decision"]]

# ------------------------------------------------------------------ decision
with st.container(border=True):
    st.badge(DECISION_TEXT[r["decision"]], icon=icon, color=color)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Similar applicants default", f"{r['band_rate']:.1%}",
              help=(f"Observed default rate in this applicant's score band on held-out data, with a 95% interval of "
                    f"{r['band_low']:.1%} to {r['band_high']:.1%}. Decisions compare this interval with the "
                    f"break-even rate of {ctx.breakeven_pd:.1%}."))
    c2.metric("95% interval", f"{r['band_low']:.0%}–{r['band_high']:.0%}",
              help=f"{r['band_low']:.1%} to {r['band_high']:.1%}")
    c3.metric("Model score", f"{r['p_default']:.1%}",
              help="The calibrated probability of default for this applicant.")
    c4.metric("Loan amount", f"{r['credit']:,.0f}")
    if r["decision"] == "approve":
        st.caption(f"The whole interval is below the {ctx.breakeven_pd:.1%} break-even default rate: lending to applicants "
                   "like this is profitable, so the loan is approved automatically.")
    elif r["decision"] == "refer":
        st.caption(f"The interval straddles the {ctx.breakeven_pd:.1%} break-even default rate: the data cannot tell whether "
                   "this loan would make or lose money, so an underwriter decides (for example after checking payslips).")
    else:
        st.caption(f"The whole interval is above the {ctx.breakeven_pd:.1%} break-even default rate: lending to applicants "
                   "like this loses money on average.")
    if r["thin_file"]:
        st.badge("Thin file: no credit bureau history", icon=":material/person_off:", color="violet")

# ------------------------------------------------------------------ reasons
st.subheader("Why")
contrib = r["contributions"]
top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])
chart_df = pd.DataFrame({
    "factor": [describe(f) for f in top.index],
    "effect": top.values,
    "direction": np.where(top.values > 0, "raised risk", "lowered risk"),
})
chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("effect:Q", title="Effect on risk (log-odds, SHAP)"),
        y=alt.Y("factor:N", sort=list(chart_df["factor"]), title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raised risk", "lowered risk"],
                                                       range=["#C44E52", "#55A868"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["factor", alt.Tooltip("effect:Q", format="+.3f"), "direction"],
    )
)
st.altair_chart(chart, alt="The ten factors that moved this applicant's risk most, up or down")

left, right = st.columns(2)


def show_value(v) -> str:
    if pd.isna(v):
        return "missing"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.2f}"
    return str(v)


def reason_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Factor": df["reason"], "Applicant's value": [show_value(v) for v in df["value"]]})


with left:
    st.markdown("**What raised the risk**")
    st.table(reason_table(r["reasons"]), hide_index=True, border="horizontal", alt="Factors that raised this applicant's risk")
with right:
    st.markdown("**What lowered the risk**")
    st.table(reason_table(r["strengths"]), hide_index=True, border="horizontal", alt="Factors that lowered this applicant's risk")

# ------------------------------------------------------------------ path to approval
if r["decision"] != "approve":
    st.subheader("Path to approval")
    options = r["options"]
    if options is None:
        st.info("No change within 50% on the levers the applicant controls (borrow less, a bigger down payment, "
                "pay down debt, lower card use) is enough here.", title="No realistic path found")
    else:
        goal = "automatic approval" if r["path_target"] == "approve" else "review by an underwriter instead of a decline"
        st.markdown(f"Smallest changes the model estimates would lead to **{goal}**:")
        st.table(
            pd.DataFrame({"Change": options["change"], "New model score": [f"{p:.1%}" for p in options["new_pd"]]}),
            hide_index=True, border="horizontal", alt="Changes that would improve this applicant's decision",
        )
        st.caption("What would change the model's decision, not a guarantee that the applicant would repay.")

# ------------------------------------------------------------------ decision letter
if r["decision"] == "decline":
    st.subheader("Decision letter")
    letter = st.session_state["letters"].get(r["applicant_id"])
    if letter is None:
        st.caption(f"Written by Claude ({MODEL}; {ESCALATION_MODEL} only if it declines the request) from the facts "
                   "above, then checked by rules and a second Claude call. About 0.1 cent per letter.")
        if st.button("Write decision letter", icon=":material/edit_note:", type="primary"):
            facts = build_facts(r["applicant_id"], "declined", r["reasons"], r["strengths"],
                                r["options"] if r["path_target"] == "approve" else None)
            usage = Usage()
            with st.spinner("Writing and checking the letter..."):
                try:
                    result = generate(get_claude_client(), facts, usage)
                except Exception as e:  # missing key, network error, ...
                    st.error(f"{type(e).__name__}: {e}", title="Could not reach Claude")
                    st.stop()
            st.session_state["letters"][r["applicant_id"]] = {"result": result, "cost": usage.cost_usd,
                                                              "escalations": usage.escalations}
            st.rerun()
    else:
        result = letter["result"]
        if result["routed_to_review"]:
            st.warning(" ".join(result["failures"]), title="Sent to a person to write")
        else:
            with st.container(border=True):
                st.text(result["text"])
            status = "passed every check" if result["passed"] else "did not pass every check: " + "; ".join(result["failures"])
            st.caption(f"Letter {status} after {result['attempts']} attempt(s) · cost ${letter['cost']:.4f}"
                       + (" · escalated to Opus" if letter["escalations"] else ""))
elif r["decision"] == "refer":
    st.info("Referred applicants get a decision from an underwriter, not an automatic letter.", title="No letter yet")

with st.expander("Known outcome (test data only)"):
    st.write("Repaid" if r["outcome"] == 0 else "Defaulted")
    st.caption("Shown for the demo. A real lender would not know this at decision time.")
