"""
A seeded opening exchange, so the app is not empty on first boot.

The lines below are the *customer's* side only. The agent's replies are not
canned here — they are produced by running the real engine against whichever
provider the session resolved, which means the transcript on screen went
through the same prompt assembly, the same window and the same timing code as
anything typed by hand, and the latency shown next to each turn is measured
rather than invented.

That distinction matters for a project whose subject is latency. A screenshot
of fabricated numbers would be worth nothing.
"""

from __future__ import annotations

from .engine import ConversationEngine

#: Opening customer lines per scenario, keyed by the persona's `scenario`
#: field. Written to trip the things a support call is graded on — a greeting,
#: an account detail, a follow-up question — so a seeded call exercises the
#: review as well as the pipeline.
DEMO_OPENINGS: dict[str, list[str]] = {
    "card_lost": [
        "Hi, I think I've lost my debit card — I can't find it anywhere since yesterday.",
        "Yes, it's Priya Raman, born the fourth of March 1991.",
        "Most of them look right, but there's a forty pound one in Leeds I didn't make.",
    ],
    "transfer_failed": [
        "Hello — I tried to send money to my landlord this morning and it failed.",
        "It was six hundred and fifty pounds, sent about nine this morning.",
        "The sort code was 20-45-11 and the account number was 7741822.",
    ],
    "account_locked": [
        "Hi, I'm locked out of my account and I've got a payment due today.",
        "It's Daniel Osei, and the last four digits are 8820.",
        "Yes, that's the right mobile — the last three are 447.",
    ],
}

#: Used when a persona's scenario is not one of the three above.
FALLBACK_OPENING = [
    "Hello, I need some help with my account please.",
    "Yes, that's me — happy to confirm whatever you need.",
]


def openings_for(persona: dict[str, object]) -> list[str]:
    """The customer lines that open a demo call for this persona."""
    scenario = str(persona.get("scenario", "")).strip().lower().replace(" ", "_")
    return DEMO_OPENINGS.get(scenario, FALLBACK_OPENING)


def seed_demo_call(engine: ConversationEngine, turns: int = 3) -> None:
    """
    Play the opening of a call through the engine.

    Does nothing if the conversation has already started, so a reload does not
    duplicate the transcript. Failures are swallowed on purpose: a seeded demo
    is a convenience, and a provider that cannot answer should leave an empty
    conversation rather than an error page.
    """
    if engine.turns:
        return
    for line in openings_for(engine.persona)[:turns]:
        try:
            # Streamed, like a real turn: a seeded call that used the
            # whole-response path would show no time-to-first-token, which is
            # the one number the page is about.
            for _ in engine.stream_respond(line):
                pass
        except Exception:
            return
