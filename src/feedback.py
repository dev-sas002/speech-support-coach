"""
The keyword scorer for a finished call.

It is a heuristic and nothing more: four regexes over the transcript, checking
that the call opened with a greeting, verified identity, showed some empathy
and reached a resolution. It cannot judge whether the advice was correct, and
it grades the agent's side of a simulated call rather than a trainee's.

Its value is that it always works — no key, no network, no model — which makes
it the floor the model-backed review in `src.analysis.call_review` falls back
to. The criteria are exposed structurally so both paths report the same four
things in the same order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern

from .state_manager import ConversationState

#: A greeting opens the call. Grouped and anchored so "this" and "shipping"
#: are not mistaken for "hi".
GREETING_RE = re.compile(r"\b(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))\b", re.I)
VERIFICATION_RE = re.compile(r"name|verify|security|dob|date of birth|account", re.I)
EMPATHY_RE = re.compile(r"sorry|understand|i can imagine|that sounds", re.I)
RESOLUTION_RE = re.compile(r"let's|we can|i will|steps|block|reissue|unlock|transfer", re.I)


@dataclass(slots=True)
class Criterion:
    """One scored expectation, with the wording used in the report."""

    key: str
    met: bool
    present_label: str
    missing_label: str

    @property
    def label(self) -> str:
        return self.present_label if self.met else self.missing_label


#: key, pattern, label when met, label when missing. `side` says whose
#: utterances the pattern runs over.
_ASSISTANT_CRITERIA: tuple[tuple[str, Pattern[str], str, str], ...] = (
    ("verification", VERIFICATION_RE, "Verification asked", "Verification missing"),
    ("empathy", EMPATHY_RE, "Empathy detected", "Empathy missing"),
    ("resolution", RESOLUTION_RE, "Resolution provided", "Resolution unclear"),
)


def criteria(state: ConversationState) -> list[Criterion]:
    """The four scored criteria for this conversation, in report order."""
    user_utterances = [t["text"] for t in state.turns if t["role"] == "user"]
    assistant_utterances = [t["text"] for t in state.turns if t["role"] == "assistant"]

    # Alternation binds looser than the anchors: the old pattern read as
    # `\bhello` OR `hi` OR `good\s+(...)\b`, so the bare `hi` matched inside
    # "this", "which" and "shipping" and almost every call scored a greeting.
    # Only the opening exchanges count — a greeting halfway through a call is
    # not an opening.
    results = [
        Criterion(
            key="greeting",
            met=any(GREETING_RE.search(u) for u in user_utterances[:2]),
            present_label="Greeting present",
            missing_label="Missing greeting",
        )
    ]
    for key, pattern, present, missing in _ASSISTANT_CRITERIA:
        results.append(
            Criterion(
                key=key,
                met=any(pattern.search(a) for a in assistant_utterances),
                present_label=present,
                missing_label=missing,
            )
        )
    return results


def evaluate(state: ConversationState) -> str:
    """The human-readable report: what the call did, then what it missed."""
    scored = criteria(state)
    lines = ["Post-run evaluation:"]
    lines += [f"✅ {c.label}" for c in scored if c.met]
    lines += [f"⚠️ {c.label}" for c in scored if not c.met]
    return "\n".join(lines)
