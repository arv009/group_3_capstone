"""Independent audit of a drafted letter: faithfulness and tone (Phase 2, Step 9; notebook 07).

The guardrails in letters.py check which factor codes a letter says it cites and scan for banned words.
They cannot see a sentence that cites the right code but says something else (Phase 1's "the car you
plan to buy" for a car the applicant already owns). This module audits the finished text instead.

Faithfulness
  1. extract_claims(): a Claude Haiku 5.5 call reads ONLY the letter, never the fact sheet, and lists
     every factual claim as {factor_text, value_quoted, direction}. It cannot read the fact sheet into
     the letter because it never sees it.
  2. match_claims() (plain Python): each claim is matched to a fact-sheet entry by word overlap,
     preferring the kind its direction implies (raises risk -> risk factor, lowers risk -> strength,
     would help -> adjusted terms). Claims the overlap cannot place go to one more small Claude call
     that only picks the closest entry (or none). A claim is SUPPORTED when it matches an entry, its
     direction agrees with the SHAP sign, and every number it quotes appears in that entry.
  3. faithfulness = supported claims / all claims; coverage = share of the top 3 risk factors the
     letter mentions. Every unsupported claim is explained in plain words.

Tone, three ways
  * VADER compound score (-1 to 1): a word lexicon tuned for short social-media text
  * TextBlob polarity (-1 to 1) and subjectivity (0 to 1): another word lexicon
  * a Claude rubric: respectful, clear and empathetic, each 1 to 5 with a one-line reason

Every Claude answer is cached on disk by a hash of its prompt (like letters.py), so re-running the
notebook or the tests costs nothing and needs no network once the cache holds the answers.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import PROCESSED_DIR
from .letters import MODEL, PRICE_IN, PRICE_OUT

AUDIT_VERSION = "a1"
EFFORT = "low"
CACHE_PATH = PROCESSED_DIR / "letter_audit_cache.json"


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------
class Claim(BaseModel):
    factor_text: str = Field(description="What the claim is about, in the letter's own words, e.g. 'credit scores from other agencies'")
    value_quoted: str = Field(description="Any number or specific value the claim quotes, e.g. '508,842' or '40%'. Empty if none")
    direction: Literal["raises_risk", "lowers_risk", "would_help", "none"] = Field(
        description="raises_risk: counted against the applicant. lowers_risk: counted in their favour. "
                    "would_help: a change or offer the letter suggests. none: a plain statement of fact")


class Claims(BaseModel):
    claims: list[Claim]


class ClaimMatch(BaseModel):
    claim_index: int
    fact_id: str = Field(description="ID of the fact the claim is about (e.g. R2), or 'none' if no fact fits")


class ClaimMatches(BaseModel):
    matches: list[ClaimMatch]


class ToneRubric(BaseModel):
    respectful: int = Field(ge=1, le=5, description="1 = rude or blaming, 5 = consistently respectful")
    respectful_reason: str
    clear: int = Field(ge=1, le=5, description="1 = confusing or full of jargon, 5 = easy for anyone to follow")
    clear_reason: str
    empathetic: int = Field(ge=1, le=5, description="1 = cold, 5 = kindly acknowledges the applicant's position")
    empathetic_reason: str


EXTRACT_SYSTEM = """You read a letter from a lender to a loan applicant and list every factual claim it makes about the applicant, their application or their loan.
A claim is a statement about a factor that counted against them, a factor that counted in their favour, a loan offer or change that could help, or a specific fact about them or their loan.
Do NOT list: the decision itself, thanks or courtesy, apologies, the offer to review the decision or to contact the lender, or statements that something is not a promise.
Keep each factor_text short and in the letter's own words."""

MATCH_SYSTEM = """You match claims from a loan letter to a list of facts. For each claim, give the ID of the fact it is about, or 'none' if no fact is about the same thing.
Only judge what the claim is ABOUT. Do not judge whether it is accurate."""

TONE_SYSTEM = """You rate the tone of a letter from a lender to a loan applicant. Score respectful, clear and empathetic from 1 to 5, with one short sentence of reason for each."""


# ---------------------------------------------------------------------------
# Cache and Claude calls
# ---------------------------------------------------------------------------
class AuditCache:
    """Disk cache of structured Claude answers, keyed by a hash of (model, version, schema, prompts).

    With client=None a cache miss raises instead of calling the API, so tests stay offline."""

    def __init__(self, path: Path | None = CACHE_PATH):
        self.path = path
        self.data: dict = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}
        self.hits = 0

    @staticmethod
    def key(system: str, user: str, schema: type[BaseModel]) -> str:
        return hashlib.sha256(f"{MODEL}|{AUDIT_VERSION}|{schema.__name__}|{system}|{user}".encode()).hexdigest()[:24]

    def parse(self, client, usage: "Usage", system: str, user: str, schema: type[BaseModel]):
        k = self.key(system, user, schema)
        if k in self.data:              # cached: free now, but still counted at what it cost to produce
            self.hits += 1
            entry = self.data[k]
            usage.add(entry["input_tokens"], entry["output_tokens"], cached=True)
            return schema.model_validate(entry["output"])
        if client is None:
            raise LookupError("Answer not cached and no Claude client given")
        response = client.messages.parse(model=MODEL, max_tokens=4000, system=system, output_config={"effort": EFFORT},
                                         messages=[{"role": "user", "content": user}], output_format=schema)
        usage.add(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined the audit request")
        self.data[k] = {"output": response.parsed_output.model_dump(),
                        "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return response.parsed_output

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")


@dataclass
class Usage:
    """Tokens and cost of the Claude calls behind the audits. cost_usd includes answers reused from the cache
    (what the audit cost to produce); live_cost_usd only the calls made in this run."""
    calls: int = 0               # live API calls
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    live_cost_usd: float = 0.0

    def add(self, tokens_in: int, tokens_out: int, cached: bool = False) -> None:
        cost = tokens_in * PRICE_IN / 1e6 + tokens_out * PRICE_OUT / 1e6
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out
        if cached:
            self.cached_calls += 1
        else:
            self.calls += 1
            self.live_cost_usd += cost

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_IN / 1e6 + self.output_tokens * PRICE_OUT / 1e6


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------
STOPWORDS = set("""a an the and or of to in on for with by from at as is are was were be been your you our we us it its this that
these those has have had not no than then so if would could some any about more less much very also just only one their
there which what how when into over under up down out such counted against favour favor helped help main reason reasons
factor factors risk well quite low high strong""".split())
KIND_OF_DIRECTION = {"raises_risk": "R", "lowers_risk": "S", "would_help": "P"}
EXPECTED_DIRECTION = {"R": "raises_risk", "S": "lowers_risk", "P": "would_help", "L": "none", "D": "none"}
SAME_WORD = {"repayment": "payment", "agency": "external", "outside": "external", "lender": "external"}
MATCH_THRESHOLD = 0.5            # share of the claim's content words that must appear in the fact
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _words(text: str) -> set[str]:
    """Content words in lower case, with a plural 's' removed so 'scores' matches 'score'."""
    words = re.findall(r"[a-z]+", text.lower())
    words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS]
    return {SAME_WORD.get(w, w) for w in words}


def overlap(claim_text: str, fact_text: str) -> float:
    a, b = _words(claim_text), _words(fact_text)
    return len(a & b) / len(a) if a else 0.0


def numbers_in(text: str) -> set[str]:
    """Every number in a text, without thousands separators: '508,842' -> '508842'."""
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


def facts_from_sheet(sheet: dict) -> dict[str, str]:
    """The fact sheet (letters.LetterInput as a dict) as numbered facts:
    R = risk factors, S = strengths, P = adjusted terms (path to approval), L = the requested loan, D = the decision."""
    facts = {f"R{i + 1}": f"{f['label']}: {f['value']}" for i, f in enumerate(sheet.get("risk_factors", []))}
    facts |= {f"S{i + 1}": f"{f['label']}: {f['value']}" for i, f in enumerate(sheet.get("strengths", []))}
    if t := sheet.get("adjusted_terms"):
        facts["P1"] = (f"Adjusted loan (a loan on different, adjusted terms that may be possible): loan amount "
                       f"{t['loan_amount']}, regular payment {t['regular_payment']}, {t['number_of_payments']}")
    # The decision itself, worded the way the letter prompt asks each decision to be explained
    facts["D1"] = {"approve": "Decision: the application is approved",
                   "under_review": "Decision: under review; the application is close to approval, so a loan officer "
                                   "will look at it personally and may contact the applicant about adjusting the loan",
                   "decline": "Decision: declined as requested; the applicant can ask for the decision to be reviewed"
                   }.get(sheet.get("decision"), "Decision")
    facts["L1"] = (f"Requested loan: loan amount {sheet['loan_amount']}, regular payment {sheet['regular_payment']}, "
                   f"{sheet['number_of_payments']}")
    return facts


def extract_claims(client, body: str, usage: Usage, cache: AuditCache) -> list[Claim]:
    """Claude lists the letter's claims. It sees only the letter."""
    return cache.parse(client, usage, EXTRACT_SYSTEM, f"LETTER:\n{body}", Claims).claims


def match_claims(claims: list[Claim], facts: dict[str, str], client=None, usage: Usage | None = None,
                 cache: AuditCache | None = None) -> list[dict]:
    """Match each claim to a fact and decide whether it is supported. One dict per claim."""
    def score(c: Claim, fid: str) -> float:
        """Word overlap; a fact that contains every number the claim quotes is a strong match on its own."""
        quoted = numbers_in(c.value_quoted) | numbers_in(c.factor_text)
        if quoted and quoted <= numbers_in(facts[fid]):
            return 1.0
        return overlap(c.factor_text, facts[fid])

    scores = [{fid: score(c, fid) for fid in facts} for c in claims]

    def candidates(i: int) -> list[str]:
        """Facts above the threshold, best first; facts of the kind the claim's direction implies come first,
        so 'a smaller loan could be offered' lands on the adjusted terms, not on a risk factor about the loan."""
        kind = KIND_OF_DIRECTION.get(claims[i].direction)
        quoted = numbers_in(claims[i].value_quoted) | numbers_in(claims[i].factor_text)
        by_words = sorted([f for f, v in scores[i].items() if v >= MATCH_THRESHOLD and f[0] == kind],
                          key=lambda f: -scores[i][f])
        by_number = [f for f in facts if quoted and quoted <= numbers_in(facts[f])]
        rest = sorted([f for f, v in scores[i].items() if v >= MATCH_THRESHOLD], key=lambda f: -scores[i][f])
        # 1) a fact of the kind the direction implies whose words match, 2) a fact holding every quoted number,
        # 3) any other fact whose words match
        return list(dict.fromkeys(by_words + by_number + rest))

    # Give each fact to one claim where possible (claims that quote numbers may share their fact) (two claims about two different scores should land on the
    # two score facts, not both on one); claims left over may share a fact.
    matched, used = {}, set()
    for i in sorted(range(len(claims)), key=lambda i: -max(scores[i].values(), default=0)):
        cands = candidates(i)
        if cands and (numbers_in(claims[i].value_quoted) | numbers_in(claims[i].factor_text)):
            matched[i] = cands[0]           # claims that quote numbers may share a fact (amount, payment, term)
            continue
        free = [f for f in cands if f not in used]
        if free:
            matched[i] = free[0]
            used.add(free[0])
    rows = []
    for i, c in enumerate(claims):
        fid = matched.get(i) or next(iter(candidates(i)), None)
        rows.append({"claim": c.factor_text, "value": c.value_quoted, "direction": c.direction,
                     "fact_id": fid, "matched_by": "words" if fid else None})

    # Claims the word overlap could not place: one Claude call picks only WHICH fact each is about.
    unplaced = [i for i, r in enumerate(rows) if r["fact_id"] is None]
    if unplaced and cache is not None:
        listing = "\n".join(f"{fid}: {t}" for fid, t in facts.items())
        asked = "\n".join(f"{i}: {claims[i].factor_text}" for i in unplaced)
        out = cache.parse(client, usage or Usage(), MATCH_SYSTEM, f"FACTS:\n{listing}\n\nCLAIMS:\n{asked}", ClaimMatches)
        for m in out.matches:
            if m.claim_index in unplaced and m.fact_id in facts:
                rows[m.claim_index].update(fact_id=m.fact_id, matched_by="claude")

    for r in rows:
        if r["fact_id"] is None and r["direction"] == "would_help" and "P1" in facts:
            r.update(fact_id="P1", matched_by="offer")      # an unplaced offer can only be the adjusted terms
        fid = r["fact_id"]
        if fid is None:
            r["supported"], r["problem"] = False, f"Not in the fact sheet: \"{r['claim']}\"."
            continue
        expected = EXPECTED_DIRECTION[fid[0]]
        quoted = numbers_in(r["value"]) | numbers_in(r["claim"])
        direction_ok = (r["direction"] in (expected, "none") or fid[0] == "D"     # the decision has no direction
                        or (fid[0] in "PL" and r["direction"] == "would_help"))
        if not direction_ok:
            said = {"raises_risk": "counted against them", "lowers_risk": "counted in their favour",
                    "would_help": "would help"}[r["direction"]]
            truth = {"raises_risk": "raised the risk", "lowers_risk": "lowered the risk",
                     "would_help": "is the suggested change", "none": "is a plain fact"}[expected]
            r["supported"] = False
            r["problem"] = f"Wrong direction: the letter says \"{r['claim']}\" {said}, but in the fact sheet it {truth}."
        elif quoted - numbers_in(facts[fid]) - numbers_in(facts["L1"]):   # restating the requested loan is fine
            r["supported"] = False
            r["problem"] = f"Wrong number: the letter quotes {r['value'] or r['claim']}, but the fact sheet says {facts[fid]}."
        else:
            r["supported"], r["problem"] = True, ""
    return rows


# ---------------------------------------------------------------------------
# Tone
# ---------------------------------------------------------------------------
def lexicon_tone(text: str) -> dict:
    """VADER compound and TextBlob polarity / subjectivity (no API calls)."""
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    blob = TextBlob(text).sentiment
    return {"vader_compound": round(SentimentIntensityAnalyzer().polarity_scores(text)["compound"], 3),
            "textblob_polarity": round(blob.polarity, 3), "textblob_subjectivity": round(blob.subjectivity, 3)}


def tone_scores(client, body: str, usage: Usage, cache: AuditCache) -> dict:
    rubric = cache.parse(client, usage, TONE_SYSTEM, f"LETTER:\n{body}", ToneRubric)
    return lexicon_tone(body) | rubric.model_dump()


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------
TONE_FLOOR = 3          # a rubric score below this on any dimension flags the letter


@dataclass
class Audit:
    claims: list[dict]
    faithfulness: float           # supported claims / all claims (1.0 when there are none)
    coverage: float | None        # share of the top 3 risk factors mentioned (None for approvals)
    tone: dict
    problems: list[str] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def n_supported(self) -> int:
        return sum(c["supported"] for c in self.claims)

    @property
    def passed(self) -> bool:
        return not self.problems

    def badge(self) -> str:
        return f"{self.n_supported} of {len(self.claims)} claims supported"

    def to_dict(self) -> dict:
        return {"badge": self.badge(), "passed": self.passed, "faithfulness": self.faithfulness,
                "coverage": self.coverage, "tone": self.tone, "problems": self.problems, "claims": self.claims,
                "cost_usd": round(self.cost_usd, 6), "audit_version": AUDIT_VERSION}


def letter_body(body: str) -> str:
    """Drop the greeting and the sign-off; they carry no claims."""
    return body.replace("Dear Applicant,", "").replace("The Lending Team", "").strip()


def audit_letter(client, body: str, fact_sheet: dict, cache: AuditCache, usage: Usage | None = None) -> Audit:
    """Faithfulness and tone for one letter. fact_sheet is the letter's LetterInput as a dict.
    Pass client=None to use cached answers only."""
    usage = usage or Usage()
    start = usage.cost_usd
    text = letter_body(body)
    facts = facts_from_sheet(fact_sheet)
    claims = match_claims(extract_claims(client, text, usage, cache), facts, client, usage, cache)
    tone = tone_scores(client, text, usage, cache)

    top3 = [k for k in facts if k.startswith("R")][:3]
    mentioned = {c["fact_id"] for c in claims if c["supported"]}
    coverage = sum(r in mentioned for r in top3) / len(top3) if top3 and fact_sheet.get("decision") != "approve" else None
    faithfulness = sum(c["supported"] for c in claims) / len(claims) if claims else 1.0

    problems = [c["problem"] for c in claims if not c["supported"]]
    low = [k for k in ("respectful", "clear", "empathetic") if tone[k] < TONE_FLOOR]
    if low:
        problems.append("Tone: " + "; ".join(f"{k} {tone[k]}/5 ({tone[k + '_reason']})" for k in low))
    return Audit(claims, faithfulness, coverage, tone, problems, usage.cost_usd - start)
