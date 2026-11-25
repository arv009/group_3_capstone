"""Tests for the AI Analyst. No API calls: run_sql and the tools run for real on the bundle, the loop runs against a
scripted fake client, and the ten test questions are graded from answers recorded in notebook 10."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secondlook import agent as A  # noqa: E402
from secondlook.analyst_eval import expected_answers, grade  # noqa: E402

RUNS = json.loads((ROOT / "tests" / "fixtures" / "analyst_runs.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- run_sql safety
@pytest.mark.parametrize("query,expected", [
    ("DROP TABLE portfolio", "Refused"),
    ("UPDATE portfolio SET pd = 0", "Refused"),
    ("SELECT 1; SELECT 2", "only one statement"),
    ("WITH x AS (SELECT 1) DELETE FROM portfolio", "Refused"),
    ("PRAGMA table_info(portfolio)", "Refused"),
    ("", "empty"),
    ("SELECT nope FROM portfolio", "SQL error: no such column"),
])
def test_unsafe_or_broken_queries_come_back_as_text(query, expected):
    assert expected in A.run_sql(query)


def test_large_results_are_capped_at_200_rows():
    out = A.run_sql("SELECT * FROM portfolio LIMIT 10000")
    lines = out.strip().splitlines()
    assert lines[-1].startswith("[Only the first 200 rows")
    assert len([l for l in lines[1:] if l and not l.startswith("[")]) == 200


def test_runaway_query_is_stopped(monkeypatch):
    monkeypatch.setattr(A, "SQL_TIMEOUT_S", 0.5)
    assert "stopped" in A.run_sql("SELECT COUNT(*) FROM portfolio a, portfolio b")


def test_trailing_semicolon_and_comments_are_fine():
    out = A.run_sql("-- count\nSELECT COUNT(*) AS n FROM portfolio;")
    assert out.splitlines() == ["n", "61503"]


def test_database_file_is_opened_read_only():
    import sqlite3
    con = sqlite3.connect(f"file:{A.DB_PATH.as_posix()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("DELETE FROM portfolio")


def test_tools_on_the_bundle():
    tb = A.Toolbox()
    p = json.loads(tb.run("path_to_approval", {"applicant_id": 209138})[0])
    assert p["status"] == "path" and p["recommended"]["new_pd"] < 0.17
    e = json.loads(tb.run("explain_applicant", {"applicant_id": 209138})[0])
    assert e["raised_risk_most"] and e["decision"] == "under_review"
    out, err = tb.run("explain_applicant", {"applicant_id": 1})
    assert err and "not one of the 300 demo applicants" in out
    f = json.loads(tb.run("fairness_summary", {"group": "gender"})[0])
    assert "never was" in f["note"] and len(f["groups"]) == 2


# ---------------------------------------------------------------- the loop, with a scripted client
def tool_use(i, name, args):
    return NS(type="tool_use", id=f"t{i}", name=name, input=args)


class FakeClient:
    """Returns the scripted responses in order and records every request."""
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.messages = self

    def create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})     # snapshot: the loop keeps appending
        content, stop = self.responses.pop(0) if self.responses else ([tool_use(99, "run_sql", {"query": "SELECT 1"})], "tool_use")
        return NS(content=content, stop_reason=stop, usage=NS(input_tokens=1000, output_tokens=100))


def test_loop_runs_a_tool_then_answers():
    client = FakeClient([([tool_use(1, "run_sql", {"query": "SELECT COUNT(*) AS n FROM portfolio"})], "tool_use"),
                         ([NS(type="text", text="There are 61,503 applicants.")], "end_turn")])
    r = A.run_agent("How many applicants?", client)
    assert r.answer == "There are 61,503 applicants." and r.stop == "answered"
    assert [s["tool"] for s in r.steps] == ["run_sql"] and "61503" in r.steps[0]["result"]
    sent = client.requests[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and sent["tool_use_id"] == "t1" and not sent["is_error"]
    assert r.cost_usd == pytest.approx(2 * (1000 * A.PRICE_IN + 100 * A.PRICE_OUT) / 1e6)


def test_loop_stops_after_six_tool_calls():
    client = FakeClient([])                         # asks for a tool forever
    r = A.run_agent("Loop forever", client)
    assert len(r.steps) == A.MAX_TOOL_CALLS
    assert r.stop == "limit"
    limit_msgs = [c for req in client.requests for m in req["messages"] if m["role"] == "user" and isinstance(m["content"], list)
                  for c in m["content"] if "limit reached" in str(c.get("content", ""))]
    assert limit_msgs, "Claude was not told the limit was reached"


def test_tool_errors_go_back_to_claude():
    client = FakeClient([([tool_use(1, "run_sql", {"query": "DROP TABLE portfolio"})], "tool_use"),
                         ([NS(type="text", text="I can only read data.")], "end_turn")])
    r = A.run_agent("Delete everything", client)
    assert r.steps[0]["is_error"] and client.requests[1]["messages"][-1]["content"][0]["is_error"]


# ---------------------------------------------------------------- the ten test questions
def test_recorded_answers_are_from_the_current_agent():
    assert RUNS["agent_version"] == A.AGENT_VERSION, "prompt or tools changed: re-run notebooks/10_ai_analyst.ipynb"


def test_at_least_nine_of_ten_questions_answered_correctly():
    results = [grade(RUNS["runs"][str(q["id"])]["answer"], q) for q in expected_answers()]
    wrong = [r for r in results if not r["correct"]]
    assert len(results) == 10 and len(wrong) <= 1, wrong


def test_every_recorded_answer_shows_its_work():
    for qid, r in RUNS["runs"].items():
        assert r["steps"], f"Q{qid} has no tool calls"
        assert r["cost_usd"] > 0


# ---------------------------------------------------------------- staying on topic, follow-ups
def test_off_topic_stops_with_the_fixed_reply_and_no_second_call():
    client = FakeClient([([tool_use(1, "off_topic", {"kind": "small_talk"})], "tool_use")])
    r = A.run_agent("How's your day?", client)
    assert r.off_topic == "small_talk" and r.answer == A.OFF_TOPIC_REPLY and r.stop == "off_topic"
    assert len(client.requests) == 1 and r.steps == []


def test_over_long_or_empty_questions_are_refused_before_any_call():
    assert A.check_question("x" * (A.MAX_QUESTION_CHARS + 1)).startswith("Please keep questions under")
    assert A.check_question("   ") == "Please type a question."
    assert A.check_question("What is the approval rate?") is None


def test_earlier_answers_carry_their_sources():
    steps = [{"tool": "run_sql", "input": {"query": "SELECT  COUNT(*)\n FROM portfolio"}},
             {"tool": "fairness_summary", "input": {"group": "gender"}}]
    msg = A.history_message("There are 61,503 applicants.", steps)
    assert msg.startswith("There are 61,503 applicants.")
    assert "[Source: computed from 2 tool calls" in msg and "run_sql: SELECT COUNT(*) FROM portfolio" in msg
    assert A.history_message("Hello", []) == "Hello"


def test_copied_source_notes_never_reach_the_page():
    content = [NS(type="text", text="The rate is 5.3%.\n\n[Source: run_sql: SELECT 1]")]
    assert A._text(content) == "The rate is 5.3%."


def test_recorded_scope_probes_are_all_classified_correctly():
    wrong = [p for p in RUNS["scope"] if bool(p["off_topic"]) != p["should_be_off_topic"]]
    assert len(RUNS["scope"]) == 9 and not wrong, wrong


def test_recorded_followups_do_not_correct_earlier_answers():
    from secondlook.analyst_eval import SELF_CORRECTION, followup_expected
    hits = [t["answer"] for t in RUNS["followup"] if SELF_CORRECTION.search(t["answer"])]
    assert not hits, hits
    exp, last = followup_expected(), RUNS["followup"][-1]["answer"]
    assert exp["persona"].lower() in last.lower() and str(exp["rate"]) in last
