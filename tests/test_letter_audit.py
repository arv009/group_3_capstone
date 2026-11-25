"""Tests for the independent letter audit. No API calls: Claude's answers for five letters are saved in
tests/fixtures/audit_cases.json (one real letter and four deliberately broken copies), and the deterministic
verdict rules are tested directly with hand-made claims."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secondlook.letter_audit import judge_claims, lexicon_tone, summarise  # noqa: E402

CASES = json.loads((ROOT / "tests" / "fixtures" / "audit_cases.json").read_text(encoding="utf-8"))

FACTS = {
    "decision": "under_review", "loan_amount": "500,000", "regular_payment": "25,000", "number_of_payments": "about 20 payments",
    "risk_factors": [{"code": "EXT_SOURCE_MEAN", "label": "Average external credit score", "value": "0.30"}],
    "strengths": [{"code": "INST_LATE_SHARE_LAST_YEAR", "label": "Share of payments late in the last year", "value": "0%"}],
    "adjusted_terms": {"loan_amount": "450,000", "regular_payment": "18,000", "number_of_payments": "about 25 payments"},
}


def claim(quote, kind="factor", presented_as="concern", value=None):
    return {"quote": quote, "kind": kind, "about": "", "value": value, "presented_as": presented_as}


def audit(case):
    c = CASES[case]
    judged = judge_claims(c["raw"]["review"]["claims"], c["raw"]["matches"], c["letter"]["fact_sheet"])
    return summarise(judged, c["letter"]["fact_sheet"], c["raw"]["review"], lexicon_tone(c["letter"]["body"]))


def test_the_real_letter_passes():
    a = audit("original")
    assert a["passed"] and a["faithfulness"] == 1.0, a["issues"]


@pytest.mark.parametrize("case,expected", [
    ("invented_detail", "car"),
    ("wrong_direction", "as a concern"),
    ("wrong_amount", "not in the fact sheet"),
    ("rude", "tone"),
])
def test_each_broken_letter_is_caught(case, expected):
    a = audit(case)
    assert not a["passed"]
    assert any(expected in i for i in a["issues"]), a["issues"]


def test_lexicon_sentiment_misses_the_rude_letter():
    # The reason the Claude rubric exists: word-list sentiment still rates the insulting letter as very positive
    assert lexicon_tone(CASES["rude"]["letter"]["body"])["vader_compound"] > 0.5
    assert audit("rude")["tone"]["respectful"] <= 2


def test_direction_value_and_fairness_rules():
    claims = [claim("your score is low", value="0.30"),                                   # correct
              claim("your score is a strength", presented_as="strength"),                 # wrong direction
              claim("your score is 0.55", value="0.55"),                                  # wrong value
              claim("your score is missing"),                                             # unfair wording
              claim("you own a boat")]                                                    # not a model reason
    matches = [{"claim_index": i, "code": "EXT_SOURCE_MEAN", "fair_description": i != 3, "note": "n"} for i in range(4)] + \
              [{"claim_index": 4, "code": None, "fair_description": True, "note": "n"}]
    v = [c["verdict"] for c in judge_claims(claims, matches, FACTS)]
    assert v[0] == "supported"
    assert "as a strength" in v[1] and "does not match" in v[2] and "misdescribes" in v[3] and "not one of the reasons" in v[4]


def test_offers_must_match_the_path():
    ok = claim("we may offer a loan of 450,000 with payments of about 18,000 over about 25 payments", kind="offer", presented_as="neutral")
    wrong = claim("we may offer a loan of 440,000", kind="offer", presented_as="neutral")
    vague = claim("a loan officer may contact you about adjusting the loan terms", kind="offer", presented_as="neutral")
    v = [c["verdict"] for c in judge_claims([ok, wrong, vague], [], FACTS)]
    assert v == ["supported", "quotes a loan amount, payment or term that is not in the fact sheet", "supported"]
    no_path = {**FACTS, "adjusted_terms": None}
    assert "no path" in judge_claims([ok], [], no_path)[0]["verdict"]
    assert judge_claims([vague], [], no_path)[0]["verdict"] == "supported"   # the v5 wording is allowed


def test_general_statements_are_not_counted():
    judged = judge_claims([claim("on-time payments show a reliable record", kind="general", presented_as="neutral")], [], FACTS)
    review = {"respectful": 5, "clear": 5, "empathetic": 5, "tone_note": ""}
    s = summarise(judged, FACTS, review, {})
    assert s["n_claims"] == 0 and s["faithfulness"] == 1.0
    assert any("makes no offer" in i for i in s["issues"])     # a path existed but the letter offered nothing


def test_percent_and_ratio_are_the_same_value():
    facts = {**FACTS, "risk_factors": [{"code": "PREV_AVG_GRANTED_TO_APPLIED", "label": "Past loans granted vs amount requested", "value": "0.97x"}]}
    c = claim("lenders granted about 97% of the amounts you requested", value="about 97%")
    m = [{"claim_index": 0, "code": "PREV_AVG_GRANTED_TO_APPLIED", "fair_description": True, "note": ""}]
    assert judge_claims([c], m, facts)[0]["verdict"] == "supported"
