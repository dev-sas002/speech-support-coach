"""
Reviewing a finished call.

Two things happen here, and they are deliberately separate.

**The compliance scan is deterministic.** Some things a bank support agent must
never do — ask for a full card number, a PIN, a full password — are matters of
fact, not judgement, and a regex finds them every time, offline, for free. A
language model asked the same question would find them *most* of the time,
which is the wrong reliability for the check that actually matters.

**The qualitative review is the model's job.** Whether the agent explained the
hold clearly, whether the tone suited a distressed caller — that is judgement,
and a keyword scorer cannot do it. When no model is reachable the review falls
back to `src.feedback`'s four keyword criteria and *says so* in `source`, so a
reader is never shown a heuristic labelled as an assessment.

The model is asked for JSON and its answer is parsed defensively: a malformed
response degrades to the heuristic rather than surfacing a parse error at the
end of a training call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from re import Pattern
from typing import Any

from .. import feedback as feedback_module
from ..providers.base import LanguageModel
from ..state_manager import ConversationState

#: The verbs that turn a mention into a request. "You'll set a new password"
#: is normal procedure; "tell me your password" is a breach, and the difference
#: is entirely in this group. Without it the scan fires on the scripted agent's
#: own correct advice, which would make it worse than useless.
_REQUEST = (
    r"\b(?:confirm|tell\s+me|provide|enter|give\s+me|read\s+out|what(?:'s|\s+is)|"
    r"need|share|say)\b"
)

#: Phrases a support agent must not use, with the reason. Matched against the
#: agent's utterances only — a customer volunteering their PIN is a different
#: (also reportable) problem, but not an agent compliance breach.
FORBIDDEN_REQUESTS: tuple[tuple[str, Pattern[str], str], ...] = (
    (
        "full_card_number",
        re.compile(
            r"\b(?:full|complete|entire|whole)\s+(?:card|debit|credit)?\s*(?:card\s+)?number\b"
            r"|\ball\s+(?:sixteen|16)\s+digits\b",
            re.I,
        ),
        "Asked for the full card number. Agents may confirm the last four digits only.",
    ),
    (
        "pin",
        re.compile(rf"{_REQUEST}[^.?!]{{0,40}}\bpin\b", re.I),
        "Asked the customer for their PIN. A PIN is never requested or confirmed on a call.",
    ),
    (
        "password",
        re.compile(rf"{_REQUEST}[^.?!]{{0,40}}\bpassword\b(?!\s+reset)", re.I),
        "Asked the customer for their password. Only a reset may be offered, never the value.",
    ),
    (
        "cvv",
        re.compile(
            rf"{_REQUEST}[^.?!]{{0,40}}\b(?:cvv|cvc|security\s+code)\b",
            re.I,
        ),
        "Asked for the card security code, which is never verified by an agent.",
    ),
    (
        "guaranteed_refund",
        re.compile(
            r"\b(?:guarantee|guaranteed|promise)\b[^.]{0,40}\b(?:refund|money back|reimburse)",
            re.I,
        ),
        "Guaranteed a refund. Dispute outcomes cannot be promised on the call.",
    ),
)


@dataclass(slots=True)
class ComplianceFlag:
    """A rule the agent's own words broke, with the words."""

    rule: str
    reason: str
    quote: str


@dataclass(slots=True)
class ReviewPoint:
    """One line of qualitative feedback."""

    label: str
    met: bool
    note: str = ""


@dataclass(slots=True)
class CallReview:
    """
    The finished review.

    `source` is load-bearing: "model" means a language model read the
    transcript, "heuristic" means four regexes did. They are not interchangeable
    and the UI labels them differently.
    """

    source: str
    score: int
    points: list[ReviewPoint] = field(default_factory=list)
    flags: list[ComplianceFlag] = field(default_factory=list)
    summary: str = ""

    @property
    def passed_compliance(self) -> bool:
        return not self.flags

    def to_markdown(self) -> str:
        origin = (
            "Reviewed by the language model."
            if self.source == "model"
            else "Reviewed by the offline keyword heuristic — no model was reachable."
        )
        lines = [f"**Score: {self.score}/100** — {origin}", ""]
        if self.summary:
            lines += [self.summary, ""]
        # List items, not bare lines: Markdown collapses single newlines, so
        # the points rendered as one run-on sentence.
        for point in self.points:
            mark = "✅" if point.met else "⚠️"
            note = f" — {point.note}" if point.note else ""
            lines.append(f"- {mark} {point.label}{note}")
        if self.flags:
            lines += ["", "**Compliance flags**", ""]
            for flag in self.flags:
                lines.append(f"- 🚩 {flag.reason}")
                lines.append(f"  > {flag.quote}")
        else:
            lines += ["", "🛡️ No compliance rules were broken."]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "score": self.score,
            "points": [{"label": p.label, "met": p.met, "note": p.note} for p in self.points],
            "flags": [{"rule": f.rule, "reason": f.reason, "quote": f.quote} for f in self.flags],
            "summary": self.summary,
        }


def scan_compliance(state: ConversationState) -> list[ComplianceFlag]:
    """Every forbidden request the agent made, in the order it made them."""
    flags: list[ComplianceFlag] = []
    for turn in state.turns:
        if turn.get("role") != "assistant":
            continue
        text = turn.get("text", "")
        for rule, pattern, reason in FORBIDDEN_REQUESTS:
            match = pattern.search(text)
            if match:
                flags.append(ComplianceFlag(rule=rule, reason=reason, quote=_excerpt(text, match)))
    return flags


def _excerpt(text: str, match: re.Match[str], window: int = 60) -> str:
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


REVIEW_SYSTEM_PROMPT = (
    "You are a quality assessor for bank contact-centre calls. You are given a "
    "transcript between a customer and a support agent. Judge only the agent.\n"
    "Reply with JSON and nothing else, in exactly this shape:\n"
    '{"score": <integer 0-100>, "summary": "<one or two sentences>", '
    '"points": [{"label": "<short expectation>", "met": true|false, '
    '"note": "<why, in one clause>"}]}\n'
    "Give between three and six points. Judge: identity verification before any "
    "account action, empathy appropriate to the situation, clarity of the next "
    "steps, and whether the call was closed properly."
)


def _transcript_text(state: ConversationState, persona_name: str) -> str:
    lines = []
    for turn in state.turns:
        who = "Customer" if turn.get("role") == "user" else persona_name
        lines.append(f"{who}: {turn.get('text', '')}")
    return "\n".join(lines)


def _parse_model_review(raw: str) -> tuple[int, str, list[ReviewPoint]] | None:
    """Pull the JSON object out of a model reply, tolerating surrounding prose."""
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    try:
        score = round(float(data.get("score", 0)))
    except (TypeError, ValueError):
        return None
    score = max(0, min(100, score))

    points: list[ReviewPoint] = []
    for item in data.get("points") or []:
        if not isinstance(item, dict) or not item.get("label"):
            continue
        points.append(
            ReviewPoint(
                label=str(item["label"]),
                met=bool(item.get("met")),
                note=str(item.get("note") or ""),
            )
        )
    if not points:
        return None
    return score, str(data.get("summary") or ""), points


def _heuristic_review(state: ConversationState) -> tuple[int, str, list[ReviewPoint]]:
    scored = feedback_module.criteria(state)
    points = [ReviewPoint(label=c.label, met=c.met) for c in scored]
    met = sum(1 for c in scored if c.met)
    score = round(100 * met / len(scored)) if scored else 0
    summary = (
        f"{met} of {len(scored)} keyword expectations were met. "
        "This is a pattern match, not an assessment of the advice given."
    )
    return score, summary, points


def review_call(
    state: ConversationState,
    llm: LanguageModel | None = None,
    persona_name: str = "Agent",
) -> CallReview:
    """
    Review a finished call, using the model when one is reachable.

    `llm` is injected rather than resolved here so tests never construct a
    provider and the caller keeps control of which one is used. Passing None,
    or passing the scripted stand-in, yields the heuristic review — the scripted
    agent replies with support dialogue, not with an assessment, so asking it to
    grade a call would produce confident nonsense.
    """
    flags = scan_compliance(state)

    if not state.turns:
        return CallReview(
            source="heuristic",
            score=0,
            points=[],
            flags=flags,
            summary="Nothing was said, so there is nothing to review.",
        )

    usable_model = llm is not None and getattr(llm, "name", "") != "scripted"
    if usable_model:
        try:
            reply = llm.complete(  # type: ignore[union-attr]
                [
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": _transcript_text(state, persona_name)},
                ]
            )
            parsed = _parse_model_review(reply.text)
        except Exception:
            # A failed review must not be the thing that breaks the call. The
            # heuristic below always produces an answer.
            parsed = None
        if parsed is not None:
            score, summary, points = parsed
            # A compliance breach is not a matter of opinion, so it caps the
            # score however well the model thought the call went.
            if flags:
                score = min(score, 50)
            return CallReview(
                source="model", score=score, points=points, flags=flags, summary=summary
            )

    score, summary, points = _heuristic_review(state)
    if flags:
        score = min(score, 50)
    return CallReview(source="heuristic", score=score, points=points, flags=flags, summary=summary)
