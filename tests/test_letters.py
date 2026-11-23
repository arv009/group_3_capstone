"""Tests for letters.py with a fake Claude client (no API key or network needed)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from secondlook.letters import (
    FOOTER, PRICE_PER_MTOK, PRICES_PER_MTOK, CitedSentence, DecisionLetter, JudgeVerdict, LetterFacts, Usage, assemble, build_facts,
    check_letter, generate,
)


def _facts(**kw):
    base = dict(applicant_id=1, decision="declined",
                reasons={"R1": "average external credit score", "R2": "share of past installments paid late"},
                strengths={"S1": "time in current job"},
                steps={"P1": "Pay 20% of the loan upfront as a bigger down payment"})
    return LetterFacts(**(base | kw))


def _letter(**kw):
    base = dict(
        subject="Your loan application",
        opening="Thank you for applying. We are not able to approve your loan at this time.",
        reasons=[CitedSentence(id="R1", sentence="Your credit scores from other sources were on the low side."),
                 CitedSentence(id="R2", sentence="Some past payments to us were made late.")],
        strengths=[CitedSentence(id="S1", sentence="Your steady job counted in your favour.")],
        steps=[CitedSentence(id="P1", sentence="Paying 20% of the loan upfront would likely change the outcome.")],
        closing="We appreciate your interest and hope to help you in future.",
    )
    return DecisionLetter(**(base | kw))


def test_good_letter_passes():
    assert check_letter(_letter(), _facts())["passed"]


def test_assembled_letter_has_our_footer_and_all_parts():
    text = assemble(_letter())
    assert FOOTER in text and "What could change the decision:" in text
    assert text.count("- ") == 1


@pytest.mark.parametrize("field,value,expected", [
    ("reasons", [CitedSentence(id="R2", sentence="Late payments."), CitedSentence(id="R1", sentence="Scores.")], "in order"),
    ("reasons", [CitedSentence(id="R1", sentence="Scores.")], "in order"),
    ("steps", [], "Steps must cite"),
    ("strengths", [CitedSentence(id="S9", sentence="Something.")], "Unknown strength"),
    ("closing", "Because of your age we cannot lend.", "protected"),
    ("closing", "If you do this you are guaranteed approval.", "Promises"),
    ("closing", "Your EXT_SOURCE_MEAN was low.", "feature code"),
    ("closing", "Your probability of default was high.", "technical"),
])
def test_bad_letters_fail(field, value, expected):
    result = check_letter(_letter(**{field: value}), _facts())
    assert not result["passed"]
    assert any(expected.lower() in f.lower() for f in result["failures"])


@pytest.mark.parametrize("closing", [
    "You are guaranteed approval if you pay more upfront.",
    "Doing this will guarantee you approval.",
    "With these changes you will definitely be approved.",
])
def test_promises_are_flagged(closing):
    assert not check_letter(_letter(closing=closing), _facts())["passed"]


@pytest.mark.parametrize("closing", [
    "These steps are not a guarantee of approval, but our model estimates they would change the outcome.",
    "We cannot guarantee the result of a new application.",
])
def test_disclaimers_are_not_flagged(closing):
    assert check_letter(_letter(closing=closing), _facts())["passed"]


def test_ordinary_words_are_not_flagged():
    letter = _letter(closing="A single payment on your older loans was late, so please manage repayments carefully.")
    assert check_letter(letter, _facts())["passed"]


def test_too_long_fails():
    assert not check_letter(_letter(closing="word " * 300), _facts())["passed"]


def test_build_facts_routes_sensitive_reasons():
    reasons = pd.DataFrame({"feature": ["EXT_SOURCE_MEAN", "NAME_FAMILY_STATUS"], "reason": ["a", "family status"]})
    strengths = pd.DataFrame({"feature": ["CNT_CHILDREN", "YEARS_EMPLOYED"], "reason": ["children", "job"]})
    facts = build_facts(7, "declined", reasons, strengths)
    assert facts.sensitive_reasons == ["NAME_FAMILY_STATUS"]
    assert facts.strengths == {"S1": "job"}  # sensitive strength dropped
    assert facts.steps == {}


class FakeClient:
    """Returns queued parsed outputs in order and records the prompts it was sent."""

    def __init__(self, outputs):
        self.outputs, self.prompts = list(outputs), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        assert kw["model"] == "claude-haiku-5-5" and "fallbacks" not in kw
        return SimpleNamespace(parsed_output=self.outputs.pop(0), stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=500), stop_details=None)


OK_VERDICT = JudgeVerdict(unsupported_statements=[], mentions_protected_attribute=False, promises_approval=False)


def test_generate_passes_first_time():
    client, usage = FakeClient([_letter(), OK_VERDICT]), Usage()
    out = generate(client, _facts(), usage)
    assert out["passed"] and out["attempts"] == 1 and usage.calls == 2
    assert usage.cost_usd == pytest.approx(2 * (1000 * PRICE_PER_MTOK["input"] + 500 * PRICE_PER_MTOK["output"]) / 1e6)


def test_generate_retries_with_feedback():
    bad = _letter(closing="You are guaranteed approval next time.")
    client = FakeClient([bad, OK_VERDICT, _letter(), OK_VERDICT])
    out = generate(client, _facts(), Usage())
    assert out["passed"] and out["attempts"] == 2
    assert "failed these checks" in client.prompts[2] and "Promises approval" in client.prompts[2]


def test_generate_fails_when_judge_finds_unsupported_claim():
    verdict = JudgeVerdict(unsupported_statements=["Mentions a missed rent payment."],
                           mentions_protected_attribute=False, promises_approval=False)
    client = FakeClient([_letter(), verdict, _letter(), verdict])
    out = generate(client, _facts(), Usage())
    assert not out["passed"] and out["attempts"] == 2
    assert any("rent" in f for f in out["failures"])


def test_generate_skips_claude_for_sensitive_reasons():
    client = FakeClient([])
    out = generate(client, _facts(sensitive_reasons=["NAME_FAMILY_STATUS"]), Usage())
    assert out["routed_to_review"] and client.prompts == []


class ScriptedClient(FakeClient):
    """Refuses on the models listed in `refuse_on`; answers with queued outputs otherwise."""

    def __init__(self, outputs, refuse_on):
        super().__init__(outputs)
        self.refuse_on, self.models = set(refuse_on), []

    def _parse(self, **kw):
        self.models.append(kw["model"])
        if kw["model"] in self.refuse_on:
            return SimpleNamespace(parsed_output=None, stop_reason="refusal",
                                   usage=SimpleNamespace(input_tokens=100, output_tokens=0),
                                   stop_details=SimpleNamespace(category="test"))
        if kw["model"] == "claude-opus-5-5":
            assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
        else:
            assert "fallbacks" not in kw
        return SimpleNamespace(parsed_output=self.outputs.pop(0), stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=500), stop_details=None)


def test_haiku_refusal_escalates_to_opus():
    usage = Usage()
    client = ScriptedClient([_letter(), OK_VERDICT], refuse_on={"claude-haiku-5-5"})
    out = generate(client, _facts(), usage)
    assert out["passed"] and not out["routed_to_review"]
    assert client.models == ["claude-haiku-5-5", "claude-opus-5-5"] * 2
    assert usage.refusals == 2 and usage.escalations == 2
    opus, haiku = PRICES_PER_MTOK["claude-opus-5-5"], PRICES_PER_MTOK["claude-haiku-5-5"]
    expected = 2 * (100 * haiku["input"] + 1000 * opus["input"] + 500 * opus["output"]) / 1e6
    assert usage.cost_usd == pytest.approx(expected)


def test_refusal_by_both_models_routes_to_human_review():
    usage = Usage()
    client = ScriptedClient([], refuse_on={"claude-haiku-5-5", "claude-opus-5-5"})
    out = generate(client, _facts(), usage)
    assert out["routed_to_review"] and not out["passed"]
    assert client.models == ["claude-haiku-5-5", "claude-opus-5-5"]
