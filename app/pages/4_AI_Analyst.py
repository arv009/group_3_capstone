"""AI Analyst: ask portfolio questions in plain English; Claude answers by writing SQL and calling SecondLook's tools."""
import json

import streamlit as st

st.set_page_config(page_title="AI Analyst · SecondLook", layout="wide")

from lib import api_key_available, footer, load_applicants, load_fairness, load_model  # noqa: E402
from secondlook import agent as AG                                                       # noqa: E402

MAX_QUESTIONS = 10            # per session, to protect the public API key
EXAMPLES = [
    "Which persona has the most repayers among the applicants we don't approve?",
    "What share of under-review applicants actually repaid?",
    "How does the approval rate of repayers differ between age bands?",
    "How many thin-file applicants were declined but repaid?",
]

st.title("AI Analyst")
st.caption("Ask a question about the 61,503 test applicants. Claude (Haiku 5.5) writes the SQL and calls SecondLook's tools: "
           "the portfolio, personas and fairness tables, applicant explanations and paths to approval. Every answer shows its work.")


@st.cache_resource(show_spinner="Loading the analyst's tools...")
def toolbox() -> AG.Toolbox:
    booster, meta, explainer = load_model()
    return AG.Toolbox(booster=booster, meta=meta, applicants=load_applicants(), explainer=explainer,
                      fairness=load_fairness())


@st.cache_resource
def claude():
    import anthropic
    return anthropic.Anthropic()


chat = st.session_state.setdefault("analyst_chat", [])        # [{"role", "content", "result"?}]
asked = sum(m["role"] == "user" for m in chat)

with st.sidebar:
    st.header("About the analyst")
    st.markdown(f"- Read-only SQL on three tables, at most {AG.MAX_ROWS} rows per query\n"
                f"- At most {AG.MAX_TOOL_CALLS} tool calls per question\n"
                f"- {MAX_QUESTIONS} questions per session")
    st.progress(asked / MAX_QUESTIONS, text=f"{asked} of {MAX_QUESTIONS} questions used")
    total = sum(m["result"]["cost_usd"] for m in chat if m.get("result"))
    st.caption(f"Cost this session: ${total:.4f}")
    if st.button("Clear the conversation", disabled=not chat):
        st.session_state["analyst_chat"] = []
        st.rerun()
    with st.expander("Table schemas"):
        st.code(AG.SCHEMA, language=None)


def show_work(result: dict) -> None:
    label = f"How this was answered: {len(result['steps'])} tool call(s) · ${result['cost_usd']:.4f} · {result['seconds']:.0f} s"
    with st.expander(label):
        if not result["steps"]:
            st.caption("No tool calls.")
        for i, s in enumerate(result["steps"], 1):
            st.markdown(f"**{i}. {s['tool']}**" + (" (error)" if s["is_error"] else ""))
            if s["tool"] == "run_sql":
                st.code(s["input"]["query"], language="sql")
            else:
                st.code(json.dumps(s["input"]), language="json")
            st.code(s["result"][:1500], language="json")
        if result.get("stopped_early"):
            st.caption(f"Stopped after {AG.MAX_TOOL_CALLS} tool calls and answered with what it had.")


for m in chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("result"):
            show_work(m["result"])

if not api_key_available():
    st.info("The AI Analyst needs a Claude API key (ANTHROPIC_API_KEY in .env locally, or in the app's secrets).")
    footer()
    st.stop()

question = None
if not chat:
    st.markdown("**Try a question:**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, use_container_width=True):
            question = ex

if asked >= MAX_QUESTIONS:
    st.warning(f"This session has used its {MAX_QUESTIONS} questions. Clear the conversation in the sidebar to start again.")
    typed = None
else:
    typed = st.chat_input("Ask about approvals, defaults, personas or fairness...")
question = typed or question

if question:
    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in chat][-8:]     # short memory of earlier turns
    with st.chat_message("assistant"):
        with st.spinner("Writing SQL and checking the numbers..."):
            try:
                r = AG.ask(claude(), question, toolbox(), history=history)
                result = {"steps": r.steps, "cost_usd": r.cost_usd, "seconds": r.seconds, "stopped_early": r.stopped_early}
                answer = r.answer or "No answer was returned."
            except Exception as e:                      # API or network error: show it, don't crash the demo
                result, answer = None, f"Could not reach Claude: {e}"
        st.markdown(answer)
        if result:
            show_work(result)
    chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer, "result": result}]
    st.rerun()

footer()
