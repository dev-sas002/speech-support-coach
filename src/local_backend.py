"""
A local stand-in for the Groq client, used when no API key is configured.

Both `ASRClient` and `LLMClient` construct `Groq(api_key=os.getenv(...))` in
their constructors, so without a key the assistant could not be started at all
— and neither could the parts worth reviewing: the conversation state machine,
the persona system, the turn logging, the latency metrics and the feedback
scoring. None of those need a hosted model.

This mimics the shape of the Groq client rather than changing the callers:
`client.chat.completions.create(...)` and
`client.audio.transcriptions.create(...)`, streaming and not. `LLMClient` and
`ASRClient` already accept an injected client for testing, so the same seam
serves the offline path.

The replies are **scripted and stage-aware**, not generated. A bank-support
training tool whose offline mode invented plausible-sounding financial advice
would be actively harmful — the trainee cannot tell which parts are real
procedure. Every line here is fixed text, chosen by which stage of the call
the conversation has reached.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

LOCAL_MODEL_NAME = "local-scripted-agent"

# Scripted support-agent turns per scenario, in call order. The stage advances
# with each assistant turn, so a conversation progresses rather than repeating.
SCRIPTS: dict[str, list[str]] = {
    "card_lost": [
        "I'm sorry to hear your card has gone missing — that's stressful, and "
        "we'll get it sorted. For security, can you confirm the name on the "
        "account and your date of birth?",
        "Thank you. I've placed a temporary block on the card, so nothing "
        "further can be charged to it. Do you recognise the most recent "
        "transactions on the account?",
        "Understood. I've raised a dispute for the transactions you don't "
        "recognise, and ordered a replacement card to your registered "
        "address. It typically arrives within five working days.",
        "Your reference for this case is CL-4417. Is there anything else I "
        "can help you with today?",
    ],
    "transfer_failed": [
        "I'm sorry the transfer didn't go through. Let's find out why. Can "
        "you confirm the amount and the date you attempted it?",
        "Thank you. I can see the payment was rejected by the receiving bank "
        "rather than declined here. The funds have not left your account. "
        "Can you confirm the sort code and account number you used?",
        "That account number is one digit short, which is why it was "
        "rejected. If you re-enter it with the full eight digits, the "
        "transfer should complete normally.",
        "I've noted the failed attempt on your account so no duplicate charge "
        "is applied. Anything else I can help with?",
    ],
    "account_locked": [
        "I understand — being locked out is frustrating. I can help you "
        "regain access. For security, can you confirm your full name and "
        "the last four digits of your account number?",
        "Thank you. The lock was applied automatically after several failed "
        "sign-in attempts. It is a protective measure and your funds are "
        "unaffected. I'll send a verification code to the mobile number on "
        "file — can you confirm the last three digits?",
        "The code is on its way. Once you enter it, the account will unlock "
        "immediately and you'll be prompted to set a new password.",
        "You're all set. Your reference for this call is AL-9082. Is there "
        "anything else I can do for you?",
    ],
}

CLOSING = (
    "Thanks for your patience today. If anything else comes up, call us back "
    "and quote your reference and we'll pick up where we left off."
)

#: Words in a persona's system prompt that identify which script to use.
SCENARIO_HINTS = {
    "card_lost": ("lost their debit card", "lost card", "card has been lost", "card lost"),
    "transfer_failed": ("transfer", "payment failed", "failed transfer"),
    "account_locked": ("locked", "account is locked", "locked out"),
}


def detect_scenario(messages: list[dict[str, str]]) -> str:
    """Identify the persona from the system prompt, defaulting to card_lost."""
    system = " ".join(m.get("content", "") for m in messages if m.get("role") == "system").lower()
    for scenario, hints in SCENARIO_HINTS.items():
        if any(hint in system for hint in hints):
            return scenario
    return "card_lost"


def count_assistant_turns(messages: list[dict[str, str]]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def scripted_reply(messages: list[dict[str, str]]) -> str:
    """The next line of the script for this conversation."""
    scenario = detect_scenario(messages)
    script = SCRIPTS.get(scenario, SCRIPTS["card_lost"])
    stage = count_assistant_turns(messages)
    if stage < len(script):
        return script[stage]
    return CLOSING


# --- Objects shaped like the Groq SDK's responses ------------------------


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Delta:
    content: str | None


@dataclass
class _StreamChoice:
    delta: _Delta


@dataclass
class _StreamChunk:
    choices: list[_StreamChoice]


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _Completion:
    choices: list[_Choice]
    model: str = LOCAL_MODEL_NAME
    usage: _Usage = field(default_factory=_Usage)


@dataclass
class _Transcription:
    text: str


class _Completions:
    def create(
        self,
        *,
        messages: list[dict[str, str]],
        model: str = LOCAL_MODEL_NAME,
        stream: bool = False,
        **_: Any,
    ):
        text = scripted_reply(messages)

        if not stream:
            prompt_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
            completion_tokens = len(text) // 4
            return _Completion(
                choices=[_Choice(message=_Message(content=text))],
                model=model,
                usage=_Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    # This was left at the dataclass default of 0 while the two
                    # components were populated, so the UI's token column read
                    # zero on every offline turn.
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

        def chunks() -> Iterator[_StreamChunk]:
            # Word by word, with a small delay, so the UI's streaming path is
            # exercised offline instead of only when someone is paying for it.
            for word in re.findall(r"\S+\s*", text):
                time.sleep(0.012)
                yield _StreamChunk(choices=[_StreamChoice(delta=_Delta(content=word))])
            yield _StreamChunk(choices=[_StreamChoice(delta=_Delta(content=None))])

        return chunks()


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class _Transcriptions:
    def create(self, **_: Any) -> _Transcription:
        # There is no local speech recogniser here. Saying so is better than
        # returning invented words and letting the rest of the pipeline treat
        # them as what the trainee said.
        return _Transcription(
            text="[speech recognition unavailable offline — set GROQ_API_KEY, or type your message]"
        )


class _Audio:
    def __init__(self) -> None:
        self.transcriptions = _Transcriptions()


class LocalGroqClient:
    """Same call shape as `groq.Groq`, no network."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.chat = _Chat()
        self.audio = _Audio()


#: Wall-clock ceiling on a single hosted request. The SDK's own default is
#: generous, and a voice turn that has already blown its latency budget is
#: better failed than left hanging with the microphone open.
DEFAULT_TIMEOUT_S = 30.0


def request_timeout_s() -> float:
    """The per-request timeout, overridable with GROQ_TIMEOUT_SECONDS."""
    import os

    raw = (os.getenv("GROQ_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return value if value > 0 else DEFAULT_TIMEOUT_S


def usable_api_key(name: str = "GROQ_API_KEY") -> str | None:
    """
    The key, or None when it is absent or an obvious placeholder.

    Copying `.env.example` leaves values like `your_groq_api_key_here` behind,
    which are non-empty and select the hosted path — producing a 401 from the
    vendor rather than anything pointing at the real cause.
    """
    import os

    value = (os.getenv(name) or "").strip()
    if not value:
        return None
    lowered = value.lower()
    placeholders = ("your_", "your-", "changeme", "xxx", "<", "api_key_here", "add_your")
    if any(marker in lowered for marker in placeholders):
        return None
    return value
