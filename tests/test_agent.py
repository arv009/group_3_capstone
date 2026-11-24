"""AI Analyst (Step 12): run_sql safety and the agent loop with a scripted Claude client (no API calls)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secondlook.agent import MAX_ROWS, Toolbox, ask, run_sql  # noqa: E402

DB = ROOT / "app" / "demo_data" / "analyst.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="analyst.db not built (python -m secondlook.analyst_db)")


@pytest.mark.parametrize("query", [
    "DROP TABLE portfolio",
    "UPDATE portfolio SET pd = 0",
    "SELECT 1; DROP TABLE portfolio",
    "SELECT * FROM portfolio; SELECT 1",
    "WITH x AS (SELECT 1) DELETE FROM portfolio",
    "PRAGMA table_info(portfolio)",
])
def test_unsafe_queries_are_refused(query):
    out = run_sql(query, DB)
    assert out.startswith("Error")
    assert json.loads(run_sql("SELECT COUNT(*) FROM portfolio", DB))["rows"] == [[61503]]   # nothing changed


def test_big_query_is_capped():
    out = json.loads(run_sql("SELECT * FROM portfolio LIMIT 10000", DB))
    assert out["row_count"] == MAX_ROWS and "truncated" in out["note"]


def test_sql_errors_come_back_as_text():
    assert run_sql("SELECT nope FROM portfolio", DB).startswith("SQL error")


def test_read_only_even_if_a_write_slips_past_the_filter(tmp_path):
    # mode=ro: SQLite itself refuses writes, independent of the keyword filter
    import sqlite3
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("CREATE TABLE t (x INTEGER)")
    con.close()


def block(kind, **kw):
    return SimpleNamespace(type=kind, **kw)


class ScriptedClient:
    """Returns queued responses and records each request."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.requests.append(kw)
        content, stop = self.responses.pop(0)
        return SimpleNamespace(content=content, stop_reason=stop, usage=SimpleNamespace(input_tokens=1000, output_tokens=100))


def sql_call(i, q):
    return block("tool_use", id=f"t{i}", name="run_sql", input={"query": q})


def test_loop_runs_tools_and_returns_the_answer():
    client = ScriptedClient([
        ([sql_call(1, "SELECT COUNT(*) FROM portfolio WHERE has_bureau_history = 0")], "tool_use"),
        ([block("text", text="There are 8,776 thin-file applicants.")], "end_turn"),
    ])
    r = ask(client, "How many thin-file applicants?", Toolbox(db_path=DB))
    assert r.answer == "There are 8,776 thin-file applicants."
    assert r.sql == ["SELECT COUNT(*) FROM portfolio WHERE has_bureau_history = 0"]
    assert json.loads(r.steps[0]["result"])["rows"] == [[8776]]
    assert r.api_calls == 2 and r.cost_usd > 0
    sent = client.requests[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and sent["tool_use_id"] == "t1"


def test_loop_stops_after_the_tool_call_limit():
    client = ScriptedClient([([sql_call(i, "SELECT 1")], "tool_use") for i in range(3)]
                            + [([block("text", text="Best answer so far.")], "end_turn")])
    r = ask(client, "q", Toolbox(db_path=DB), max_tool_calls=3)
    assert len(r.steps) == 3 and r.stopped_early and r.answer == "Best answer so far."
    assert client.requests[-1]["tool_choice"] == {"type": "none"}


def test_unknown_applicant_is_an_error_not_a_crash():
    import pandas as pd
    tb = Toolbox(applicants=pd.DataFrame({"SK_ID_CURR": [1]}), db_path=DB)
    text, is_error = tb.run("explain_applicant", {"applicant_id": 999})
    assert is_error and "not among the 300" in text
