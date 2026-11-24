"""AI Analyst: a manager asks a question in plain English, Claude answers by writing SQL and calling our tools
(Phase 2, Step 12; app page 4 and notebook 10).

The agent loop is written by hand (no framework) so every step can be explained:
  1. send the question, the table schemas and the tool definitions to Claude
  2. if Claude asks for tools, run each one and send the results back
  3. repeat until Claude gives a final answer, or stop after MAX_TOOL_CALLS tool calls and ask for an answer

Tools
  run_sql(query)              one read-only SELECT on app/demo_data/analyst.db (portfolio, personas, fairness)
  explain_applicant(id)       PD, decision and SHAP reasons for one of the 300 applicants in the app bundle
  path_to_approval(id)        the re-scored path to approval for one of those applicants (notebook 06)
  fairness_summary(group)     group metrics and bootstrap gaps from the fairness audit (notebook 08)

Safety of run_sql: the file is opened read-only (mode=ro), only a single SELECT (or WITH ... SELECT) is
accepted, results are capped at MAX_ROWS rows, and every error goes back to Claude as text so it can
fix its own query. The public app also limits each session to 10 questions to protect the API key.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .analyst_db import DB_PATH, SCHEMA
from .letters import MODEL, PRICE_IN, PRICE_OUT

MAX_TOOL_CALLS = 6
MAX_ROWS = 200
EFFORT = "low"

SYSTEM_PROMPT = f"""You are SecondLook's AI Analyst. Loan managers ask you questions about a portfolio of 61,503 held-out loan applicants scored by a LightGBM credit model. Answer with numbers you computed with the tools, never from memory or guesses.

Data (SQLite, read-only):
{SCHEMA}

Definitions: "approved" means decision = 'approve' (pd < 0.17). "Not approved" means under_review or decline. "Thin file" means has_bureau_history = 0. "Repayers" means defaulted = 0. Rates are shares between 0 and 1; report them as percentages with one decimal.

How to work:
- Use run_sql for counts, rates and comparisons. Write one SELECT per call; use ROUND() and LIMIT. If a query fails, read the error and fix it.
- Use explain_applicant or path_to_approval only when asked about one specific applicant.
- Use fairness_summary for fairness questions that need bootstrap confidence intervals; the fairness table has the point estimates.
- Report only numbers that a tool returned. Never interpolate, extrapolate or estimate between them: the path tool tests 5% steps, so quote the tested step.
- Answer in 2-5 short sentences of plain English for a manager: the number first, then what it means. Mention a caveat only if it matters.
- Gender and age are in the data only for fairness questions. Never suggest using them to make lending decisions."""

TOOLS = [
    {
        "name": "run_sql",
        "description": ("Run ONE read-only SQLite SELECT query on the tables portfolio, personas and fairness (schemas "
                        f"in the system prompt) and get the rows back as JSON. At most {MAX_ROWS} rows are returned. "
                        "Errors come back as text so you can fix the query."),
        "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "A single SELECT statement"}},
                         "required": ["query"]},
    },
    {
        "name": "explain_applicant",
        "description": ("Predicted risk (PD), decision and the main SHAP reasons (factors that raised and lowered the risk) "
                        "for one applicant. Only the 300 applicants in the app bundle have full features; others return an error."),
        "input_schema": {"type": "object", "properties": {"applicant_id": {"type": "integer"}}, "required": ["applicant_id"]},
    },
    {
        "name": "path_to_approval",
        "description": ("The smallest realistic change to one applicant's loan (borrow up to 40% less or up to 50% more payments) "
                        "that brings the re-scored PD under the cutoff, or 'no_path'. Only for the 300 applicants in the app bundle."),
        "input_schema": {"type": "object", "properties": {"applicant_id": {"type": "integer"}}, "required": ["applicant_id"]},
    },
    {
        "name": "fairness_summary",
        "description": ("Fairness audit for one grouping: approval rate, default rate, approval rate of repayers (equal "
                        "opportunity), calibration and four-fifths ratio per group, plus bootstrap 95% intervals for the gaps."),
        "input_schema": {"type": "object", "properties": {"group": {"type": "string", "enum": ["gender", "age_band", "credit_file"]}},
                         "required": ["group"]},
    },
]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|trigger)\b", re.I)


def run_sql(query: str, db_path: Path = DB_PATH, max_rows: int = MAX_ROWS) -> str:
    """Run one read-only SELECT. Returns JSON rows, or an error message (as text) that Claude can act on."""
    q = query.strip().rstrip(";").strip()
    if ";" in q:
        return "Error: only one statement is allowed per call."
    if not re.match(r"^(select|with)\b", q, re.I):
        return "Error: only SELECT queries are allowed."
    if FORBIDDEN.search(q):
        return "Error: the query contains a write or schema keyword; only reading is allowed."
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)       # read-only at the database level too
        try:
            cur = con.execute(q)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(max_rows + 1)
        finally:
            con.close()
    except sqlite3.Error as e:
        return f"SQL error: {e}"
    note = ""
    if len(rows) > max_rows:
        rows, note = rows[:max_rows], f" (truncated to the first {max_rows} rows: aggregate or add LIMIT)"
    return json.dumps({"columns": cols, "rows": rows, "row_count": len(rows), "note": note.strip()}, default=str)


@dataclass
class Toolbox:
    """The non-SQL tools need the bundle's model and applicants; the app passes its cached copies in."""
    booster: object = None
    meta: dict | None = None
    applicants: object = None        # DataFrame with SK_ID_CURR and the model features
    explainer: object = None
    fairness: dict | None = None
    db_path: Path = DB_PATH

    @classmethod
    def from_bundle(cls, bundle: Path | None = None) -> "Toolbox":
        import lightgbm as lgb
        import pandas as pd
        from .explain import Explainer
        bundle = bundle or DB_PATH.parent
        booster = lgb.Booster(model_str=(bundle / "model.txt").read_bytes().decode("utf-8").replace("\r\n", "\n"))
        return cls(booster=booster, meta=json.loads((bundle / "model_meta.json").read_text()),
                   applicants=pd.read_parquet(bundle / "applicants.parquet"), explainer=Explainer(booster),
                   fairness=json.loads((bundle / "fairness.json").read_text()), db_path=bundle / "analyst.db")

    def _row(self, applicant_id: int):
        row = self.applicants[self.applicants["SK_ID_CURR"] == int(applicant_id)]
        if row.empty:
            raise KeyError(f"Applicant {applicant_id} is not among the 300 applicants with full features in the app bundle. "
                           "Use run_sql on the portfolio table for their PD and decision.")
        return row

    def explain_applicant(self, applicant_id: int) -> str:
        from . import modeling as M
        from .letters import decision_for
        row = self._row(applicant_id)
        exp = self.explainer.explain(M.prepare_X(row, self.meta["features"], self.meta["categorical"], self.meta["categories"]))
        top = lambda t: [{"factor": l, "value": v, "type": c} for l, v, c in zip(t["label"], t["value"], t["category"])][:5]
        return json.dumps({"applicant_id": int(applicant_id), "pd": round(exp["pd"], 4),
                           "decision": decision_for(exp["pd"], self.meta["cutoff"]), "actually_defaulted": int(row["TARGET"].iloc[0]),
                           "raised_risk": top(exp["raises"]), "lowered_risk": top(exp["lowers"])})

    def path_to_approval(self, applicant_id: int) -> str:
        from .path import path_to_approval
        row = self._row(applicant_id)
        p = path_to_approval(row, self.booster, self.meta, self.meta["cutoff"])
        keep = lambda o: None if o is None else {k: round(float(o[k]), 4) for k in
                                                 ("cut", "term_mult", "loan_amount", "regular_payment", "n_payments", "pd")}
        return json.dumps({"status": p["status"], "current": keep(p["current"]), "recommended": keep(p["recommended"]),
                           "borrow_less_only": keep(p["borrow_less"]), "longer_term_only": keep(p["longer_term"])})

    def fairness_summary(self, group: str) -> str:
        if group not in self.fairness["groups"]:
            return f"Error: group must be one of {list(self.fairness['groups'])}."
        gaps = [g for g in self.fairness["gaps"] if g["group"] == group]
        return json.dumps({"cutoff": self.fairness["cutoff"], "groups": self.fairness["groups"][group], "gaps_with_95ci": gaps},
                          default=str)

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Run one tool call. Returns (result text, is_error)."""
        try:
            if name == "run_sql":
                out = run_sql(args["query"], self.db_path)
                return out, out.startswith(("Error", "SQL error"))
            if name == "explain_applicant":
                return self.explain_applicant(args["applicant_id"]), False
            if name == "path_to_approval":
                return self.path_to_approval(args["applicant_id"]), False
            if name == "fairness_summary":
                out = self.fairness_summary(args["group"])
                return out, out.startswith("Error")
            return f"Error: unknown tool {name}", True
        except Exception as e:                       # never crash the loop: tell Claude what went wrong
            return f"Error: {e}", True


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    answer: str
    steps: list[dict] = field(default_factory=list)     # one per tool call: {tool, input, result, is_error}
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    seconds: float = 0.0
    stopped_early: bool = False

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_IN / 1e6 + self.output_tokens * PRICE_OUT / 1e6

    @property
    def sql(self) -> list[str]:
        return [s["input"]["query"] for s in self.steps if s["tool"] == "run_sql"]


def ask(client, question: str, toolbox: Toolbox, history: list[dict] | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS) -> AgentResult:
    """Answer one question. history: earlier turns as [{"role": "user"|"assistant", "content": str}, ...]."""
    start = time.time()
    messages = list(history or []) + [{"role": "user", "content": question}]
    result = AgentResult(answer="")
    while True:
        out_of_calls = len(result.steps) >= max_tool_calls
        response = client.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM_PROMPT, messages=messages, tools=TOOLS,
            output_config={"effort": EFFORT},
            # after the last allowed tool call, Claude must answer with what it has
            **({"tool_choice": {"type": "none"}} if out_of_calls else {}),
        )
        result.api_calls += 1
        result.input_tokens += response.usage.input_tokens
        result.output_tokens += response.usage.output_tokens
        if response.stop_reason == "refusal":
            result.answer = "Claude declined to answer this question."
            break
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_calls:
            result.answer = "".join(b.text for b in response.content if b.type == "text").strip()
            result.stopped_early = out_of_calls
            break
        # Run every tool Claude asked for and send all the results back in one user message
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for call in tool_calls:
            text, is_error = toolbox.run(call.name, call.input)
            result.steps.append({"tool": call.name, "input": call.input, "result": text[:4000], "is_error": is_error})
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    result.seconds = round(time.time() - start, 1)
    return result
