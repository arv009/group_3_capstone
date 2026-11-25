"""Independent audit of a drafted letter: is every claim faithful to the model's reasons, and is the tone right?

The rule checks in letters.py verify WHICH factors a letter cites, not WHAT it says about them. A letter can
cite the right factor and still misdescribe it ("the car you plan to buy" for a car the applicant already owns).
This module adds a second, independent check:

  1. Blind review (Claude): reads ONLY the letter, never the fact sheet, and lists every factual claim with the
     value it quotes and whether the letter presents it as a concern or a strength. It also rates the tone.
  2. Matching (Claude): links each claim to a fact-sheet factor (or none) and says whether the wording is a fair
     description of that factor's label and value.
  3. Verdict (Python, deterministic): a claim is supported only if it matches a factor the model really gave,
     in the right direction (raised vs lowered risk), with numbers that agree with the fact sheet, and is fairly
     worded. Loan details and offers must quote amounts from the fact sheet.
  4. Tone, three ways: VADER and TextBlob (lexicon-based, run locally) and the Claude rubric from step 1.

faithfulness = supported claims / checkable claims. General statements ("a loan officer will review it",
"on-time payments show a reliable record") are allowed by the prompt and not counted.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Literal

from pydantic import BaseModel, Field

from .config import PROCESSED_DIR
from .letters import EFFORT, MODEL, PRICE_IN, PRICE_OUT, _client

AUDIT_VERSION = "a4"
CACHE_PATH = PROCESSED_DIR / "audit_cache.json"
MIN_RESPECTFUL = 4          # Claude tone rubric: a letter scoring below this on respect is flagged

# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------
class Claim(BaseModel):
    quote: str = Field(description="The exact words from the letter that make this claim")
    kind: Literal["factor", "loan_terms", "offer", "general"] = Field(
        description="factor: something about the applicant's record, scores, history or a loan ratio given as part "
                    "of the decision; loan_terms: the requested amount, payment or number of payments; offer: new "
                    "terms the lender may offer; general: policy, process or general lender reasoning")
    about: str = Field(description="What the claim is about, in a few plain words")
    value: str | None = Field(description="The number or value the letter states for it, exactly as written, or null")
    presented_as: Literal["concern", "strength", "neutral"] = Field(
        description="concern if the letter presents it as counting against the applicant, strength if in their favour")


class LetterReview(BaseModel):
    claims: list[Claim]
    respectful: int = Field(description="1-5: courteous and never blaming or judgemental")
    clear: int = Field(description="1-5: plain words, easy for anyone to follow")
    empathetic: int = Field(description="1-5: acknowledges the applicant's situation and next steps")
    tone_note: str = Field(description="One sentence on the tone")


class Match(BaseModel):
    # Field order matters: the note (reasoning) is written before the verdict
    claim_index: int
    code: str | None = Field(description="Code of the fact-sheet entry the claim describes, or null if none")
    note: str = Field(description="One short sentence comparing the claim with that entry")
    fair_description: bool = Field(description="True if the wording fairly describes that entry's label and value")


class Matches(BaseModel):
    matches: list[Match]


REVIEW_SYSTEM = """You audit decision letters that a consumer lender sends to loan applicants. You see only the letter.

1. List every factual claim the letter makes about the applicant or their loan, one entry per distinct fact. Copy the exact words as the quote. Classify each one:
- factor: something about the applicant's record, credit scores, payment history, past loans, other lenders, documents, or a loan ratio, given as part of the decision.
- loan_terms: the amount, regular payment or number of payments they asked for.
- offer: new terms the lender may be able to offer.
- general: policy, process, next steps, or a general statement of why lenders look at something.
Give the value exactly as the letter states it (or null), and whether the letter presents the fact as a concern (counting against the applicant), a strength (in their favour) or neutral. Judge this from how the letter frames it (for example, a fact listed in a paragraph about strengths is a strength), not from your own view of whether the fact sounds good or bad.

2. Rate the tone from 1 (poor) to 5 (excellent): respectful (courteous, never blaming), clear (plain words), empathetic (acknowledges the applicant's situation and what they can do next). Add one sentence on the tone."""

MATCH_SYSTEM = """You check claims taken from a loan decision letter against the fact sheet the letter had to be based on.

The fact sheet lists the factors that raised the applicant's estimated risk (section risk_factors), the factors that lowered it (section strengths), the loan they applied for (LOAN_REQUESTED) and, if any, new terms that would be approved (LOAN_OFFER).

For each claim, give the code of the entry it describes, or null if it describes none of them. A claim that names a group of factors (for example "your external credit scores") belongs to the closest listed factor. Write a one-sentence note, then say whether the wording is a fair description of that entry's label and value.

It is NOT fair if it changes the meaning: for example it says information is missing or limited when the fact is a low score, says a car is being bought when the fact is a car already owned, gives a different number, claims a threshold, comparison or cause the fact does not state, or overstates it (for example "consistently healthy" from a single score).

It IS fair to use an everyday word that matches the section (a risk factor described as low, weak or on the low side; a strength described as good or solid), to leave out the exact value, or to add one short general reason why lenders look at that kind of factor (for example "on-time payments show a reliable repayment record" or "a long-held ID document points to stability"). Do not judge whether it counts for or against the applicant; that is checked separately."""

# ---------------------------------------------------------------------------
# Deterministic verdict
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
LOAN_CODES = {"LOAN_REQUESTED", "LOAN_OFFER"}


def matcher_entries(fact_sheet: dict) -> list[dict]:
    """Everything the matcher may link a claim to: the factors plus the loan applied for and any offer."""
    entries = [{"code": f["code"], "label": f["label"], "value": f["value"], "section": s}
               for s in ("risk_factors", "strengths") for f in fact_sheet.get(s, [])]
    entries.append({"code": "LOAN_REQUESTED", "label": "The loan applied for: amount, regular payment, number of payments (if the decision is approve, this is also the loan approved)",
                    "value": f"{fact_sheet.get('loan_amount')}, {fact_sheet.get('regular_payment')}, {fact_sheet.get('number_of_payments')}",
                    "section": "loan"})
    if adj := fact_sheet.get("adjusted_terms"):
        entries.append({"code": "LOAN_OFFER", "label": "New terms that would be approved: amount, regular payment, number of payments",
                        "value": f"{adj['loan_amount']}, {adj['regular_payment']}, {adj['number_of_payments']}", "section": "loan"})
    return entries


def _numbers(text: str | None) -> list[float]:
    out = []
    for n in _NUM.findall(text or ""):
        try:
            out.append(float(n.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def _close(a: float, b: float) -> bool:
    """Same number allowing for rounding in prose: 0.65 vs 0.65, 9 vs 9.20 days, 11% vs 11%."""
    tol = 0.011 if abs(b) < 1 else max(0.5, 0.01 * abs(b))
    return abs(a - b) <= tol


def _same_value(a: float, b: float) -> bool:
    """Like _close, but a share may be written either way: '97%' in a letter equals '0.97x' on the fact sheet."""
    return _close(a, b) or _close(a / 100, b) or _close(a * 100, b)


def judge_claims(claims: list[dict], matches: list[dict], fact_sheet: dict) -> list[dict]:
    """Attach a verdict ('supported' | an issue string | 'not checked') to every claim."""
    risk = {f["code"]: f for f in fact_sheet.get("risk_factors", [])}
    strong = {f["code"]: f for f in fact_sheet.get("strengths", [])}
    adj = fact_sheet.get("adjusted_terms") or {}
    loan_text = " ".join([fact_sheet.get("loan_amount", ""), fact_sheet.get("regular_payment", ""),
                          fact_sheet.get("number_of_payments", "")])
    by_index = {m["claim_index"]: m for m in matches}
    judged = []
    loan_numbers = _numbers(loan_text + " " + " ".join(adj.values()))
    for i, c in enumerate(claims):
        verdict = "supported"
        m = by_index.get(i, {})
        code = m.get("code")
        if c["kind"] == "general":
            verdict = "not checked"
        elif c["kind"] == "offer" or code in LOAN_CODES or (c["kind"] == "loan_terms" and not code):
            # Loan details and offers: every number must be one of the fact sheet's loan figures
            amounts = [n for n in _numbers(c["quote"]) if abs(n) >= 1000]
            if c["kind"] == "offer" and amounts and not adj:
                verdict = "offers specific new terms, but the model found no path to approval"
            elif any(not any(_close(n, a) for a in loan_numbers) for n in _numbers(c["quote"]) if abs(n) >= 1):
                verdict = "quotes a loan amount, payment or term that is not in the fact sheet"
            elif code in LOAN_CODES and not m.get("fair_description", True):
                verdict = f"misdescribes the loan terms: {m.get('note', '')}"
        else:
            fact = risk.get(code) or strong.get(code)
            if fact is None:
                verdict = "not one of the reasons the model gave"
            elif code in risk and c["presented_as"] == "strength":
                verdict = f"presents a factor that raised risk ({fact['label']}) as a strength"
            elif code in strong and c["presented_as"] == "concern":
                verdict = f"presents a strength ({fact['label']}) as a concern"
            elif c.get("value") and _numbers(c["value"]) and _numbers(fact["value"]) and \
                    not any(_same_value(n, f) for n in _numbers(c["value"]) for f in _numbers(fact["value"]) + loan_numbers):
                verdict = f"value '{c['value']}' does not match the fact sheet ('{fact['value']}')"
            elif not m.get("fair_description", True):
                verdict = f"misdescribes {fact['label']}: {m.get('note', '')}"
        judged.append({**c, "code": by_index.get(i, {}).get("code"), "verdict": verdict})
    return judged


def summarise(judged: list[dict], fact_sheet: dict, review: dict, tone_lexicon: dict) -> dict:
    checkable = [c for c in judged if c["verdict"] != "not checked"]
    supported = [c for c in checkable if c["verdict"] == "supported"]
    issues = [f"\"{c['quote']}\": {c['verdict']}" for c in checkable if c["verdict"] != "supported"]
    top3 = [f["code"] for f in fact_sheet.get("risk_factors", [])[:3]]
    mentioned = {c["code"] for c in judged if c.get("code")}
    if fact_sheet.get("adjusted_terms") and not any(c["kind"] == "offer" for c in judged):
        issues.append("the model found a path to approval, but the letter makes no offer")
    review = {**review, **{k: min(5, max(1, int(review[k]))) for k in ("respectful", "clear", "empathetic")}}
    if review["respectful"] < MIN_RESPECTFUL:
        issues.append(f"tone: respect scored {review['respectful']}/5 ({review['tone_note']})")
    return {
        "faithfulness": len(supported) / len(checkable) if checkable else 1.0,
        "n_claims": len(checkable), "n_supported": len(supported),
        "coverage_top3": (sum(c in mentioned for c in top3) / len(top3)) if top3 and fact_sheet.get("decision") != "approve" else None,
        "passed": not issues, "issues": issues, "claims": judged,
        "tone": {"respectful": review["respectful"], "clear": review["clear"], "empathetic": review["empathetic"],
                 "note": review["tone_note"], **tone_lexicon},
    }


def lexicon_tone(text: str) -> dict:
    """VADER compound (-1..1) and TextBlob polarity (-1..1) / subjectivity (0..1): word-list sentiment, no API."""
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    blob = TextBlob(text).sentiment
    return {"vader_compound": SentimentIntensityAnalyzer().polarity_scores(text)["compound"],
            "textblob_polarity": blob.polarity, "textblob_subjectivity": blob.subjectivity}


# ---------------------------------------------------------------------------
# Claude calls with cache
# ---------------------------------------------------------------------------
def _key(letter: dict) -> str:
    raw = json.dumps([letter["subject"], letter["body"], letter["fact_sheet"], AUDIT_VERSION, MODEL], sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _parse(client, system: str, content: str, schema):
    r = client.messages.parse(model=MODEL, max_tokens=8000, system=system, output_config={"effort": EFFORT},
                              messages=[{"role": "user", "content": content}], output_format=schema)
    if r.stop_reason == "refusal":
        raise RuntimeError("Claude declined to audit this letter")
    return r.parsed_output, r.usage.input_tokens, r.usage.output_tokens


def audit_letter(letter: dict, use_cache: bool = True, cache_path=None) -> dict:
    """Audit one letter dict (subject, body, fact_sheet), e.g. a result of letters.generate_letter."""
    path = cache_path or CACHE_PATH
    key = _key(letter)
    cache = _load(path) if use_cache else {}
    if key in cache:
        raw = cache[key]
    else:
        client = _client()
        start = time.time()
        review, i1, o1 = _parse(client, REVIEW_SYSTEM, f"Letter:\nSubject: {letter['subject']}\n\n{letter['body']}", LetterReview)
        factors = matcher_entries(letter["fact_sheet"])
        claims = [c.model_dump() for c in review.claims]
        to_match = [{"claim_index": i, "quote": c["quote"], "about": c["about"], "value": c["value"]}
                    for i, c in enumerate(claims) if c["kind"] in ("factor", "loan_terms")]
        matches, i2, o2 = (Matches(matches=[]), 0, 0) if not to_match else _parse(
            client, MATCH_SYSTEM, json.dumps({"fact_sheet": factors, "claims": to_match}, indent=2), Matches)
        tokens_in, tokens_out = i1 + i2, o1 + o2
        raw = {"review": review.model_dump(), "matches": [m.model_dump() for m in matches.matches],
               "input_tokens": tokens_in, "output_tokens": tokens_out,
               "cost_usd": round(tokens_in * PRICE_IN / 1e6 + tokens_out * PRICE_OUT / 1e6, 6),
               "seconds": round(time.time() - start, 1), "model": MODEL, "audit_version": AUDIT_VERSION}
        if use_cache:
            cache = _load(path)
            cache[key] = raw
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    judged = judge_claims(raw["review"]["claims"], raw["matches"], letter["fact_sheet"])
    result = summarise(judged, letter["fact_sheet"], raw["review"], lexicon_tone(letter["body"]))
    return {**result, "raw": raw, "cost_usd": raw["cost_usd"], "model": raw["model"], "audit_version": raw["audit_version"]}
