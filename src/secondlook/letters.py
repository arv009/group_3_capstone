"""Plain-language decision letters written by Claude, checked against the model (notebook 05).

Design: Claude never sees raw applicant data. It receives only the FACTS the model produced:
the decision, the top SHAP reasons (as plain-language phrases with IDs R1, R2, ...), what worked
in the applicant's favour (S1, S2, ...) and the path-to-approval options (P1, P2, ...).

Claude returns a STRUCTURED letter (structured outputs): every reason and every step it writes
must cite one of those IDs. The final text is assembled from those parts, plus a fixed footer
written by us (not by Claude) offering a human review.

Two layers of checks, then one retry with feedback if anything fails:
  1. Deterministic checks (check_letter): valid IDs, all main reasons covered, no protected
     attributes, no promises of approval, no raw feature codes, length limit.
  2. Faithfulness judge (judge_letter): a second Claude call reads ONLY the facts and the final
     letter, and lists any statement that the facts do not support.

The API key is read from the environment or .env (ANTHROPIC_API_KEY); never hard-code it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, Field

MODEL = "claude-haiku-5-5"            # every request goes here first: fast and low-cost
ESCALATION_MODEL = "claude-opus-5-5"   # only if Haiku declines (Haiku 5.5 has no server-side fallback)
EFFORT = "medium"
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # lets Opus 5.5 fall back server-side if it declines too
# USD per million tokens. Haiku 5.5 rates apply to prompts of 100K tokens or fewer (ours are about 2-3K).
PRICES_PER_MTOK = {"claude-haiku-5-5": {"input": 0.10, "output": 0.50},
                   "claude-opus-5-5": {"input": 4.00, "output": 20.00}}
PRICE_PER_MTOK = PRICES_PER_MTOK[MODEL]

FOOTER = ("You can ask for a person to review this decision. Reply to this letter or contact our "
          "lending team, and an underwriter will look at your application again.")


# ----------------------------------------------------------------------------- facts


# Model inputs that relate to protected characteristics (marital/family status). A letter must not
# cite them, and hiding a real reason would be dishonest, so these applicants go to a human reviewer.
SENSITIVE_FEATURES = {"NAME_FAMILY_STATUS", "CNT_CHILDREN"}


@dataclass
class LetterFacts:
    """Everything Claude is allowed to say, with IDs it must cite."""
    applicant_id: int
    decision: str                       # "declined" or "approved"
    reasons: dict[str, str]             # {"R1": "average external credit score", ...} raised risk
    strengths: dict[str, str]           # {"S1": "...", ...} lowered risk
    steps: dict[str, str] = field(default_factory=dict)  # {"P1": "Pay 20% of the loan upfront ...", ...}
    sensitive_reasons: list[str] = field(default_factory=list)  # protected-related features among the main reasons

    def as_prompt(self) -> str:
        lines = [f"Decision: the application was {self.decision}."]
        lines.append("Factors that RAISED the assessed risk (most important first):")
        lines += [f"  {k}: {v}" for k, v in self.reasons.items()]
        if self.strengths:
            lines.append("Factors that LOWERED the assessed risk:")
            lines += [f"  {k}: {v}" for k, v in self.strengths.items()]
        if self.steps:
            lines.append("Changes that our model estimates would lead to approval if the applicant re-applied:")
            lines += [f"  {k}: {v}" for k, v in self.steps.items()]
        else:
            lines.append("No small change to the loan was found that would lead to approval.")
        return "\n".join(lines)


def build_facts(applicant_id: int, decision: str, reasons: pd.DataFrame, strengths: pd.DataFrame,
                options: pd.DataFrame | None = None, n_reasons: int = 3, n_strengths: int = 2) -> LetterFacts:
    """From explain.top_reasons / explain.path_to_approval outputs to LetterFacts."""
    steps = {}
    if options is not None and len(options):
        steps = {f"P{i + 1}": c for i, c in enumerate(options["change"].tolist())}
    top = reasons.head(n_reasons)
    good = strengths[~strengths["feature"].isin(SENSITIVE_FEATURES)].head(n_strengths)
    return LetterFacts(
        applicant_id=int(applicant_id), decision=decision,
        reasons={f"R{i + 1}": r for i, r in enumerate(top["reason"].tolist())},
        strengths={f"S{i + 1}": r for i, r in enumerate(good["reason"].tolist())},
        steps=steps,
        sensitive_reasons=[f for f in top["feature"] if f in SENSITIVE_FEATURES],
    )


# ----------------------------------------------------------------------------- structured output


class CitedSentence(BaseModel):
    id: str = Field(description="The ID of the fact this sentence explains, e.g. R1, S2 or P1.")
    sentence: str = Field(description="One plain-language sentence addressed to the applicant.")


class DecisionLetter(BaseModel):
    subject: str = Field(description="Short subject line.")
    opening: str = Field(description="1-2 sentences stating the decision. No reasons here.")
    reasons: list[CitedSentence] = Field(description="One sentence per R fact, in order.")
    strengths: list[CitedSentence] = Field(description="Optional: sentences about S facts.")
    steps: list[CitedSentence] = Field(description="One sentence per P fact. Empty if there are no P facts.")
    closing: str = Field(description="1-2 sentences. Courteous, no new facts.")


class JudgeVerdict(BaseModel):
    unsupported_statements: list[str] = Field(
        description="Statements in the letter that are not supported by the facts. Empty if none.")
    mentions_protected_attribute: bool = Field(
        description="True if the letter refers to age, gender, sex, marital or family status, race, religion, or nationality.")
    promises_approval: bool = Field(description="True if the letter guarantees approval if the applicant makes a change.")


SYSTEM = """You write decision letters for SecondLook, a consumer lender, explaining a credit decision to the applicant.

Rules:
- Use only the facts provided. Do not add any reason, number, policy or detail that is not in the facts.
- Write one reason sentence for each R fact, in the order given, citing its ID. Say in everyday words what the factor is and that it counted against the application.
- Do not explain WHY a factor affects risk, and do not guess at the applicant's situation, history or behaviour (for example, never write "which suggests past borrowing has not gone smoothly"). The facts say only which factors raised or lowered the risk.
- Strengths (S facts) are optional; mention at most two, citing their IDs.
- Write one step sentence for each P fact, citing its ID. Present steps as options that our model estimates would lead to approval, never as a guarantee.
- Never mention or hint at age, gender, sex, marital or family status, race, religion or nationality, even if a fact seems to relate to them.
- Do not mention probabilities, scores, cutoffs, "the model", "SHAP" or any technical term.
- Plain, respectful language a 14-year-old could follow. About 120-200 words in total. No greeting or sign-off (they are added separately)."""

JUDGE_SYSTEM = """You audit credit decision letters for faithfulness. You are given the FACTS a letter was allowed to use and the LETTER itself.
List every statement in the letter that is not supported by the facts (a reason, number, policy or claim the facts do not contain).
General courtesy, the decision itself, and the standard offer of a human review are supported. Rephrasing a fact in everyday words is supported.
Also report whether the letter mentions a protected attribute and whether it promises approval."""


# ----------------------------------------------------------------------------- assembly and checks


def assemble(letter: DecisionLetter) -> str:
    """Final letter text. The footer is ours, not Claude's."""
    parts = [f"Subject: {letter.subject}", "", "Dear applicant,", "", letter.opening, ""]
    parts += [s.sentence for s in letter.reasons]
    if letter.strengths:
        parts += [""] + [s.sentence for s in letter.strengths]
    if letter.steps:
        parts += ["", "What could change the decision:"] + [f"- {s.sentence}" for s in letter.steps]
    parts += ["", letter.closing, "", FOOTER, "", "SecondLook lending team"]
    return "\n".join(parts)


# Unambiguous protected-attribute words. Subtler mentions are left to the judge, so that ordinary
# phrases like "a single payment" or "your older loans" are not flagged.
PROTECTED = re.compile(r"\b(age|aged|years old|gender|male|female|sex|married|marital|divorced|widow\w*|"
                       r"children|race|racial|religio\w*|nationality|ethnic\w*)\b", re.I)
# Only POSITIVE promises. "This is not a guarantee" is the wording we ask for, so the bare word is not flagged.
PROMISE = re.compile(r"\b(you (are|will be) guaranteed|guarantees? (you )?(an? )?approval|guaranteed (an? )?approval|"
                     r"(you )?will (definitely |certainly )?be approved|assured of approval)\b", re.I)
FEATURE_CODE = re.compile(r"\b[A-Z]{2,}(_[A-Z0-9]+)+\b")
TECHNICAL = re.compile(r"\b(shap|probability|cutoff|threshold|log-odds|algorithm|lightgbm)\b", re.I)


def check_letter(letter: DecisionLetter, facts: LetterFacts, max_words: int = 260) -> dict:
    """Deterministic checks. Returns {"passed": bool, "failures": [...]}"""
    failures = []
    reason_ids = [s.id for s in letter.reasons]
    if reason_ids != list(facts.reasons):
        failures.append(f"Reasons must cite {list(facts.reasons)} in order; got {reason_ids}.")
    bad_s = [s.id for s in letter.strengths if s.id not in facts.strengths]
    if bad_s:
        failures.append(f"Unknown strength IDs: {bad_s}.")
    step_ids = [s.id for s in letter.steps]
    if step_ids != list(facts.steps):
        failures.append(f"Steps must cite {list(facts.steps)} in order; got {step_ids}.")

    written = " ".join([letter.subject, letter.opening, letter.closing]
                       + [s.sentence for s in letter.reasons + letter.strengths + letter.steps])
    if m := PROTECTED.search(written):
        failures.append(f"Mentions a protected attribute ('{m.group(0)}').")
    if m := PROMISE.search(written):
        failures.append(f"Promises approval ('{m.group(0)}').")
    if m := FEATURE_CODE.search(written):
        failures.append(f"Contains a raw feature code ('{m.group(0)}').")
    if m := TECHNICAL.search(written):
        failures.append(f"Uses a technical term ('{m.group(0)}').")
    n_words = len(written.split())
    if n_words > max_words:
        failures.append(f"Too long ({n_words} words; limit {max_words}).")
    return {"passed": not failures, "failures": failures, "words": n_words}


# ----------------------------------------------------------------------------- Claude calls


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    refusals: int = 0       # requests Haiku declined
    escalations: int = 0    # of those, sent on to Opus
    cost_usd: float = 0.0   # priced per call at the rate of the model that answered it

    def add(self, response, model: str = MODEL) -> None:
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        price = PRICES_PER_MTOK[model]
        self.cost_usd += (response.usage.input_tokens * price["input"] + response.usage.output_tokens * price["output"]) / 1e6


class RefusalError(RuntimeError):
    pass


def _parse(client, usage: Usage, system: str, user: str, output_format):
    """Ask Claude Haiku 5.5; if it declines, send the same request to Claude Opus 5.5.

    Haiku 5.5 has no server-side fallback (a `fallbacks` parameter returns 400), so the escalation is
    done here. Opus 5.5 keeps `fallbacks="default"`, so the API can still fall back once more server-side.
    Only if that whole chain declines is RefusalError raised (the applicant then goes to a person)."""
    request = dict(max_tokens=16000, system=system, messages=[{"role": "user", "content": user}],
                   output_format=output_format, output_config={"effort": EFFORT})
    response = client.beta.messages.parse(model=MODEL, **request)
    usage.add(response, MODEL)
    if response.stop_reason == "refusal":
        usage.refusals += 1
        usage.escalations += 1
        response = client.beta.messages.parse(model=ESCALATION_MODEL, betas=[FALLBACK_BETA], fallbacks="default", **request)
        usage.add(response, ESCALATION_MODEL)
        if response.stop_reason == "refusal":
            raise RefusalError(f"Request declined by {MODEL} and {ESCALATION_MODEL} "
                               f"({getattr(response.stop_details, 'category', None)}).")
    return response.parsed_output


def write_letter(client, facts: LetterFacts, usage: Usage, feedback: list[str] | None = None) -> DecisionLetter:
    user = "FACTS:\n" + facts.as_prompt()
    if feedback:
        user += "\n\nYour previous draft failed these checks. Fix them:\n" + "\n".join(f"- {f}" for f in feedback)
    return _parse(client, usage, SYSTEM, user, DecisionLetter)


def judge_letter(client, facts: LetterFacts, text: str, usage: Usage) -> JudgeVerdict:
    user = f"FACTS:\n{facts.as_prompt()}\n\nLETTER:\n{text}"
    return _parse(client, usage, JUDGE_SYSTEM, user, JudgeVerdict)


def generate(client, facts: LetterFacts, usage: Usage, max_attempts: int = 2) -> dict:
    """Write -> check -> judge, retrying once with the failures as feedback.

    Applicants whose main reasons include a protected-related feature are routed to a human
    reviewer without calling Claude; so is any applicant whose request Claude declines."""
    if facts.sensitive_reasons:
        return {"applicant_id": facts.applicant_id, "attempts": 0, "letter": None, "text": None, "checks": None,
                "verdict": None, "passed": False, "routed_to_review": True,
                "failures": [f"Main reasons include {facts.sensitive_reasons}; route to a human reviewer."]}
    feedback, result, history = None, {}, []
    for attempt in range(1, max_attempts + 1):
        try:
            letter = write_letter(client, facts, usage, feedback)
            text = assemble(letter)
            checks = check_letter(letter, facts)
            verdict = judge_letter(client, facts, text, usage)
        except RefusalError as err:  # declined by Haiku and by Opus: a person writes this letter
            return {"applicant_id": facts.applicant_id, "attempts": attempt, "letter": None, "text": None,
                    "checks": None, "verdict": None, "passed": False, "routed_to_review": True,
                    "failures": [f"Claude declined the request: {err}"], "history": history}
        judge_failures = [f"Unsupported statement: {s}" for s in verdict.unsupported_statements]
        if verdict.mentions_protected_attribute:
            judge_failures.append("Judge: mentions a protected attribute.")
        if verdict.promises_approval:
            judge_failures.append("Judge: promises approval.")
        history.append(checks["failures"] + judge_failures)
        result = {"applicant_id": facts.applicant_id, "attempts": attempt, "letter": letter, "text": text,
                  "checks": checks, "verdict": verdict, "routed_to_review": False,
                  "passed": checks["passed"] and not judge_failures,
                  "failures": checks["failures"] + judge_failures,
                  "history": history}  # failures on every attempt, first attempt first
        if result["passed"]:
            break
        feedback = result["failures"]
    return result
