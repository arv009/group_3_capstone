"""Applicant review: what a loan officer sees for one applicant."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Applicant Review · SecondLook", layout="wide")

from lib import (DECISION_STYLE, L, P, api_key_available, audit_live_letter, decision_badge,  # noqa: E402
                 explain_applicant, footer, load_applicants, load_letter_audits, load_letters, load_model, money,
                 path_for, score_applicants)

_, meta, _ = load_model()
scored = score_applicants()
letters = load_letters()
audits = load_letter_audits()
cutoff = meta["cutoff"]


def letter_key(sid: int, decision: str) -> str:
    return f"{sid}|{decision}|{L.PROMPT_VERSION}|{L.MODEL}"


scored["HAS_LETTER"] = [letter_key(s, d) in letters for s, d in zip(scored["SK_ID_CURR"], scored["DECISION"])]

st.title("Applicant Review")
st.caption("The loan officer's view of one application: risk score, decision, the reasons behind it, and a drafted letter.")

# ---------------------------------------------------------------- choose applicant
with st.sidebar:
    st.header("Choose an applicant")
    group = st.radio("Decision group", ["Under review", "Decline", "Approve", "All"], index=0,
                     help="'Under review' = just above the cutoff: SecondLook's focus.")
    only_letters = st.toggle("Only applicants with a drafted letter", value=True)
    pool = scored
    if group != "All":
        pool = pool[pool["DECISION"] == {"Under review": "under_review", "Decline": "decline", "Approve": "approve"}[group]]
    if only_letters:
        pool = pool[pool["HAS_LETTER"]]
    pool = pool.sort_values("PD")
    if pool.empty:
        st.warning("No applicants match. Turn off 'Only applicants with a drafted letter'.")
        st.stop()
    labels = {int(r.SK_ID_CURR): f"#{int(r.SK_ID_CURR)} · risk {r.PD:.1%}{' · letter ready' if r.HAS_LETTER else ''}" for r in pool.itertuples()}
    sid = st.selectbox("Applicant", list(labels), format_func=labels.get)
    st.caption(f"{len(pool)} applicants in this view. \"letter ready\" = letter already drafted.")

row = scored.set_index("SK_ID_CURR").loc[sid]
exp = explain_applicant(sid)
decision = row["DECISION"]

# ---------------------------------------------------------------- summary
st.markdown(f"### Applicant #{sid} &nbsp; {decision_badge(decision)}", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Predicted risk (PD)", f"{exp['pd']:.1%}", delta=f"{(exp['pd'] - cutoff) * 100:+.1f} pts vs cutoff {cutoff:.0%}", delta_color="inverse")
c2.metric("Loan amount", money(row["AMT_CREDIT"]))
c3.metric("Regular payment", money(row["AMT_ANNUITY"]))
c4.metric("Number of payments", f"{row['CREDIT_TERM']:.0f}")
c5.metric("Credit bureau record", "Yes" if row["HAS_BUREAU_HISTORY"] else "Thin file", help="Thin file = no record at any credit bureau")

if decision == "under_review":
    st.warning(f"**Borderline:** risk is within 5 points above the {cutoff:.0%} cutoff. In the test data, about 80% of applicants "
               "in this band repaid. Review the reasons below before deciding.")

# ---------------------------------------------------------------- reasons
st.subheader("Why the model gave this score")
left, right = st.columns([1, 1])
with left:
    top = exp["all"].head(12).iloc[::-1]
    names = [f"{'[sensitive] ' if c == 'sensitive' else ''}{l} = {v}" for l, v, c in zip(top["label"], top["value"], top["category"])]
    fig = go.Figure(go.Bar(x=top["shap"], y=names, orientation="h",
                           marker_color=["#C44E52" if s > 0 else "#4C72B0" for s in top["shap"]]))
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(height=460, margin=dict(t=10, l=10), xaxis_title="Push on risk (red = raises, blue = lowers)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each bar is one factor's contribution to this applicant's score (SHAP). They add up exactly to the model's output. "
               "[sensitive] = sensitive characteristic: shown for transparency, never cited in letters or suggested as a change.")
with right:
    cat_icon = {"actionable": "loan term", "external score": "external score", "credit history": "history",
                "profile": "profile", "sensitive": "sensitive"}
    def table(t):
        return pd.DataFrame({"Factor": t["label"], "Value": t["value"], "Type": t["category"].map(cat_icon)})
    cfg = {"Factor": st.column_config.TextColumn(width="medium"), "Value": st.column_config.TextColumn(width="small"),
           "Type": st.column_config.TextColumn(width="small")}
    st.markdown("**Raised risk most**")
    st.dataframe(table(exp["raises"]), hide_index=True, use_container_width=True, column_config=cfg)
    st.markdown("**Lowered risk most (strengths)**")
    st.dataframe(table(exp["lowers"]), hide_index=True, use_container_width=True, column_config=cfg)

# ---------------------------------------------------------------- path to approval
path = path_for(sid) if decision != "approve" else None


def describe(o) -> str:
    parts = ([f"{o['cut']:.0%} less"] if o["cut"] > 0 else []) + \
            ([f"{o['term_mult'] - 1:.0%} more payments"] if o["term_mult"] > 1 else [])
    return " + ".join(parts)


if path is not None:
    st.subheader("Path to approval")
    if path["status"] == "path":
        opts = [("Current loan", path["current"])]
        for label, kind in [("Recommended", "recommended"), ("Borrow less only", "borrow_less"),
                            ("Longer term only", "longer_term")]:
            o = path[kind]
            if o and not any(o["cut"] == x["cut"] and o["term_mult"] == x["term_mult"] for _, x in opts):
                opts.append((label, o))
        st.dataframe(pd.DataFrame({
            "Option": [lab for lab, _ in opts],
            "Change": ["none"] + [describe(o) for _, o in opts[1:]],
            "Loan amount": [money(o["loan_amount"]) for _, o in opts],
            "Regular payment": [money(o["regular_payment"]) for _, o in opts],
            "Payments": [f"{o['n_payments']:.0f}" for _, o in opts],
            "Predicted risk": [f"{o['pd']:.1%}" for _, o in opts],
            "Decision": [DECISION_STYLE[L.decision_for(o["pd"], cutoff)][0] for _, o in opts],
        }), hide_index=True, use_container_width=True)
        st.caption(f"Each option is re-scored by the same model and must be at least {P.MARGIN * 100:.1f} points under the "
                   f"{cutoff:.0%} cutoff. Only the loan changes: up to 40% less, or up to 50% more payments (at most "
                   f"{P.TERM_CAP:.0f}). Recommended = the smallest change. Payment = amount ÷ payments, before interest. "
                   "The model reflects past patterns, not a guarantee: the officer decides.")
    else:
        st.info("No realistic change to the loan (up to 40% less, or up to 50% more payments) brings the risk under "
                "the cutoff. The risk comes mainly from the applicant's history, not the loan.")

# ---------------------------------------------------------------- letter
st.subheader("Draft letter to the applicant")
apps = load_applicants().set_index("SK_ID_CURR")
li = L.build_letter_input(sid, exp, apps.loc[sid], cutoff, path=path)
key = letter_key(sid, decision)
letter = letters.get(key) or st.session_state.get(f"letter_{key}")

if letter is None:
    st.write("No letter drafted yet for this applicant.")
    if api_key_available():
        if st.button("Draft letter with Claude", type="primary"):
            with st.spinner("Claude is drafting the letter..."):
                try:
                    letter = L.generate_letter(li, use_cache=False)
                    st.session_state[f"letter_{key}"] = letter
                    with st.spinner("Auditing the letter (faithfulness and tone)..."):
                        st.session_state[f"audit_{key}"] = audit_live_letter(letter)
                    st.rerun()
                except Exception as e:     # API, network or refusal: show it, don't crash the demo
                    st.error(f"Could not draft the letter: {e}")
    else:
        st.info("Live drafting needs a Claude API key (add ANTHROPIC_API_KEY to the app's secrets). "
                "Applicants marked \"letter ready\" have letters drafted in advance.")

if letter:
    lc, rc = st.columns([3, 1])
    with lc:
        st.markdown(f"**Subject:** {letter['subject']}")
        st.text_area("Letter (editable before sending)", letter["body"], height=360, key=f"body_{key}")
    with rc:
        if letter["checks_passed"]:
            st.success("All guardrail checks passed")
        else:
            st.error("Checks flagged issues:\n- " + "\n- ".join(letter["issues"]))
        from lib import label_of
        st.markdown("**Reasons cited**  \n" + ("\n".join(f"- {label_of(c)}" for c in letter["reasons_cited"]) or "none (approval)"))
        st.markdown("**Strengths cited**  \n" + "\n".join(f"- {label_of(c)}" for c in letter["strengths_cited"]))
        audit = audits.get(key) or st.session_state.get(f"audit_{key}")
        if audit:
            st.markdown("**Letter audit**")
            badge = st.success if audit["passed"] else st.warning
            badge(f"{audit['badge']}" + (f" · covers {audit['coverage']:.0%} of the top 3 reasons"
                                         if audit.get("coverage") is not None else ""))
            t = audit["tone"]
            st.caption(f"Tone (Claude rubric, 1-5): respectful {t['respectful']}, clear {t['clear']}, "
                       f"empathetic {t['empathetic']} · VADER {t['vader_compound']:+.2f} · "
                       f"TextBlob {t['textblob_polarity']:+.2f}")
            for problem in audit["problems"]:
                st.caption(f"Check: {problem}")
        st.caption(f"Drafted by {letter['model']} · prompt {letter['prompt_version']} · {letter['attempts']} attempt(s) · "
                   f"${letter['cost_usd']:.5f}" + (f" · audit ${audit['cost_usd']:.5f}" if audit else ""))
        st.download_button("Download letter (.txt)", f"Subject: {letter['subject']}\n\n{letter['body']}", file_name=f"letter_{sid}.txt")

# ---------------------------------------------------------------- officer decision
st.subheader("Loan officer decision")
log = st.session_state.setdefault("decision_log", [])
d1, d2 = st.columns([1, 2])
with d1:
    choice = st.radio("Your decision", ["Approve", "Approve with adjusted terms", "Decline", "Request more information"],
                      index=0 if decision == "approve" else 3)
with d2:
    note = st.text_area("Note for the file", placeholder="e.g. Late payments in the last year were during a job change; now employed 3 years.")
if st.button("Record decision"):
    log.append({"Applicant": sid, "Model decision": DECISION_STYLE[decision][0], "PD": f"{exp['pd']:.1%}", "Officer decision": choice, "Note": note})
    st.success("Decision recorded for this session.")
if log:
    st.dataframe(pd.DataFrame(log), hide_index=True, use_container_width=True)

with st.expander("Reveal actual outcome (test data only)"):
    st.write("This applicant **defaulted**." if row["TARGET"] == 1 else "This applicant **repaid**.")
    st.caption("Only possible because this is historical test data. In real use the outcome is unknown at decision time.")
footer()
