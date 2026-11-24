"""Letter audit (Step 9): four deliberately bad letters must be flagged, a good one must pass.

Claude's answers are cached in tests/fixtures/letter_audit_cache.json, so these tests run offline
with no API key. After changing a prompt in letter_audit.py, refresh the cache once with
    python tests/test_letter_audit.py --record        (about 15 Haiku calls, well under one cent)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secondlook.letter_audit import AuditCache, Claim, audit_letter, facts_from_sheet, lexicon_tone, match_claims  # noqa: E402

CACHE = Path(__file__).parent / "fixtures" / "letter_audit_cache.json"

SHEET = {
    "applicant_id": 1, "decision": "under_review",
    "loan_amount": "500,000", "regular_payment": "25,000", "number_of_payments": "about 20 payments",
    "risk_factors": [{"code": "EXT_SOURCE_MEAN", "label": "Average external credit score", "value": "0.31"},
                     {"code": "INST_LATE_SHARE", "label": "Share of past payments made late", "value": "40%"},
                     {"code": "CREDIT_TO_GOODS", "label": "Loan amount vs price of the goods", "value": "1.25"}],
    "strengths": [{"code": "BUR_N_ACTIVE_LOANS", "label": "Active loans at other lenders", "value": "0"}],
    "adjusted_terms": {"loan_amount": "425,000", "regular_payment": "21,250", "number_of_payments": "about 20 payments"},
}

GOOD = """Dear Applicant,

Thank you for applying. Your application is close, so a loan officer will look at it personally.

The main concerns are your credit scores from other agencies, some late payments in the past, and a loan amount that is high compared with the price of the goods. On the plus side, you have no active loans with other lenders.

A loan officer may be able to offer a loan of 425,000 with regular payments of about 21,250. This is a possibility, not a promise.

The Lending Team"""

BAD = {
    "invented detail": GOOD.replace("you have no active loans with other lenders.",
                                    "you have no active loans with other lenders, and the car you plan to buy also helps."),
    "wrong direction": GOOD.replace("On the plus side, you have no active loans with other lenders.",
                                    "Your active loans with other lenders also counted against you."),
    "wrong amount": GOOD.replace("425,000", "400,000"),
    "rude": """Dear Applicant,

Your application is not good enough. Your credit is poor, you pay late and you want far too much money for what you are buying. Having no other loans is the only thing going for you.

Take 425,000 with payments of 21,250 or don't bother. We are not going to waste more time on this.

The Lending Team""",
}
EXPECTED = {"invented detail": "Not in the fact sheet", "wrong direction": "Wrong direction",
            "wrong amount": "Wrong number", "rude": "Tone"}


@pytest.fixture(scope="module")
def cache():
    if not CACHE.exists():
        pytest.skip("No cached Claude answers: run `python tests/test_letter_audit.py --record` once")
    return AuditCache(CACHE)


def test_good_letter_passes(cache):
    audit = audit_letter(None, GOOD, SHEET, cache)
    assert audit.passed, audit.problems
    assert audit.faithfulness == 1.0 and audit.coverage == 1.0


@pytest.mark.parametrize("kind", list(BAD))
def test_bad_letter_is_flagged(cache, kind):
    audit = audit_letter(None, BAD[kind], SHEET, cache)
    assert not audit.passed
    assert any(p.startswith(EXPECTED[kind]) for p in audit.problems), audit.problems


def test_matching_rules_without_claude():
    facts = facts_from_sheet(SHEET)
    claims = [Claim(factor_text="credit scores from other agencies", value_quoted="", direction="raises_risk"),
              Claim(factor_text="active loans with other lenders", value_quoted="", direction="raises_risk"),
              Claim(factor_text="a loan of 400,000", value_quoted="400,000", direction="would_help"),
              Claim(factor_text="missed rent payment", value_quoted="", direction="raises_risk")]
    rows = match_claims(claims, facts)          # no cache: claims the words can't place stay unplaced
    assert [r["fact_id"] for r in rows] == ["R1", "S1", "P1", None]
    assert [r["supported"] for r in rows] == [True, False, False, False]


def test_lexicons_rank_the_rude_letter_lower():
    assert lexicon_tone(BAD["rude"])["vader_compound"] < lexicon_tone(GOOD)["vader_compound"]


if __name__ == "__main__" and "--record" in sys.argv:
    import anthropic
    from dotenv import load_dotenv

    from secondlook.letter_audit import Usage

    load_dotenv(ROOT / ".env")
    client, usage, c = anthropic.Anthropic(), Usage(), AuditCache(CACHE)
    for name, text in [("good", GOOD), *BAD.items()]:
        a = audit_letter(client, text, SHEET, c, usage)
        print(f"{name}: {a.badge()} | {a.problems}")
    c.save()
    print(f"{usage.calls} live calls, ${usage.live_cost_usd:.4f}")
