"""Tests for the rule-based letter guardrails (no API calls). Run: pytest tests/"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secondlook.letters import Letter, LetterInput, check_letter, decision_for  # noqa: E402

FILLER = "This is a sentence about the application. " * 10     # keeps letters inside the 40-140 word range


def make_input(decision="decline"):
    return LetterInput(
        applicant_id=1, decision=decision, loan_amount="500,000", regular_payment="25,000", number_of_payments="about 20 payments",
        risk_factors=[{"code": "EXT_SOURCE_MEAN", "label": "Average external credit score", "value": "0.30"},
                      {"code": "CREDIT_TO_GOODS", "label": "Loan amount vs price of goods", "value": "1.42x"}],
        strengths=[{"code": "BUR_DEBT_TO_CREDIT", "label": "Debt vs credit limit at other lenders", "value": "0%"}],
    )


def make_letter(text, reasons=("EXT_SOURCE_MEAN",), strengths=("BUR_DEBT_TO_CREDIT",), subject="Your loan application decision"):
    return Letter(subject=subject, body=FILLER + text, reasons_cited=list(reasons), strengths_cited=list(strengths))


def test_clean_letter_passes():
    assert check_letter(make_letter("We are unable to approve your application."), make_input()) == []


@pytest.mark.parametrize("text,should_flag", [
    ("The loan amount is higher than the price of the goods.", False),   # true statement of a ratio factor
    ("The loan is lower than your income.", False),
    ("The term is longer than some of your past loans.", True),          # invented comparison
    ("Your past amount is similar to this loan.", True),
    ("Your car is 4 years old.", False),                                 # car age is a real model factor
    ("As you get older, things change.", True),                          # personal characteristic
    ("Because you are married, we approved less.", True),
    ("Our model estimated your risk.", False),                           # 'model' alone is not in the jargon list
    ("Your SHAP values were high.", True),
    ("A SecondLook review was done.", True),                             # internal tool name
    ("We are happy to take a second look if you ask.", False),           # ordinary phrase
])
def test_guardrails(text, should_flag):
    issues = check_letter(make_letter(text), make_input())
    assert bool(issues) == should_flag, issues


def test_cited_codes_must_come_from_fact_sheet():
    issues = check_letter(make_letter("Unable to approve.", reasons=("AGE_YEARS",)), make_input())
    assert any("not in the fact sheet" in i for i in issues)
    assert any("sensitive" in i for i in issues)


def test_decline_needs_a_reason_and_no_approval_tone():
    issues = check_letter(make_letter("Congratulations, you are approved!", reasons=()), make_input("decline"))
    assert any("no reason" in i for i in issues)
    assert any("tone" in i for i in issues)


def test_approval_letter_must_not_list_risks():
    issues = check_letter(make_letter("Your application has been approved."), make_input("approve"))
    assert any("approval letter lists risk factors" in i for i in issues)


def test_decision_bands():
    assert decision_for(0.10, 0.17) == "approve"
    assert decision_for(0.19, 0.17) == "under_review"
    assert decision_for(0.30, 0.17) == "decline"


def test_amounts_must_come_from_the_fact_sheet():
    assert check_letter(make_letter("You asked for a loan of 500,000 with payments of 25,000."), make_input()) == []
    issues = check_letter(make_letter("You asked for a loan of 550,000."), make_input())
    assert any("amounts that are not in the fact sheet" in i for i in issues)


def test_adjusted_terms_must_be_offered_exactly():
    li = make_input("under_review")
    li.adjusted_terms = {"loan_amount": "450,000", "regular_payment": "18,000", "number_of_payments": "about 25 payments"}
    ok = "A loan officer may be able to offer a loan of 450,000 with payments of about 18,000 over about 25 payments."
    assert check_letter(make_letter(ok), li) == []
    issues = check_letter(make_letter("A loan officer may be able to offer a smaller loan."), li)
    assert any("does not offer the adjusted terms" in i for i in issues)
    issues = check_letter(make_letter("We may offer a loan of 440,000 with payments of about 18,000."), li)
    assert any("amounts that are not in the fact sheet" in i for i in issues)


@pytest.mark.parametrize("text,should_flag", [
    ("A loan on different terms may be possible, and a loan officer can confirm this.", True),
    ("A smaller loan on other terms may be available if you contact us.", True),
    ("We may be able to offer you a smaller amount.", True),
    ("A loan of 500,000 with regular payments of about 25,000 may be possible.", True),
    ("We may be able to offer a loan on other terms, so please contact us.", True),
    ("You can reply to this letter to ask us to review the decision.", False),
])
def test_no_offer_without_adjusted_terms(text, should_flag):
    """Found by the notebook 07 audit: decline letters hinted at other terms when no path to approval existed."""
    issues = check_letter(make_letter(text), make_input())
    assert any("adjusted terms" in i for i in issues) == should_flag
