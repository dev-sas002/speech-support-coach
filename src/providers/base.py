"""
The provider contracts.

A voice turn is three exchangeable steps — transcribe, think, speak — and the
only thing the rest of the system needs from each is a narrow call shape. These
protocols are that shape. Everything above them (the conversation engine, the
UI, the CLI) is written against the protocol, never against a vendor SDK, so
swapping Groq for another host, or Kokoro for another synthesiser, is a
registration plus an environment variable rather than an edit to the turn loop.

The protocols are deliberately small. `Transcriber` takes WAV bytes because
that is the one audio format every hosted recogniser accepts; `LanguageModel`
has both a whole-response and a streaming call because a voice interface needs
the streaming one and a batch evaluation needs the other.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

Message = dict[str, str]


class ProviderUnavailable(RuntimeError):
    """
    The provider exists but cannot do the work here.

    Distinct from a provider that is simply unknown (`ProviderNotRegistered`):
    this one is registered and selected, it just has no credentials, no model
    or no hardware. Callers degrade rather than crash.
    """


class ProviderNotRegistered(KeyError):
    """A provider name was requested that nothing has registered."""


@dataclass(slots=True)
class LLMReply:
    """A complete model response plus what it cost to get it."""

    text: str
    latency_ms: float
    usage: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int | None:
        if not self.usage:
            return None
        value = self.usage.get("total_tokens")
        return int(value) if value is not None else None


@dataclass(slots=True)
class Transcript:
    """What the recogniser heard, and how long it took to say so."""

    text: str
    latency_ms: float
    audio_seconds: float = 0.0


@dataclass(slots=True)
class ProviderStatus:
    """
    A provider's identity and readiness, for display.

    The UI shows this so a reviewer can see *why* the app is answering from a
    script instead of a hosted model, instead of guessing from the replies.
    """

    kind: str
    requested: str
    resolved: str
    ready: bool
    detail: str = ""


@runtime_checkable
class LanguageModel(Protocol):
    """Turns a message list into an assistant reply."""

    name: str

    def complete(self, messages: list[Message]) -> LLMReply:
        """The whole reply, once it is finished."""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Text fragments as they arrive. Yields nothing if the reply is empty."""

    def status(self) -> ProviderStatus:
        """Whether this provider can currently answer, and how."""


@runtime_checkable
class Transcriber(Protocol):
    """Turns WAV bytes into text."""

    name: str

    def transcribe(self, wav_bytes: bytes) -> Transcript: ...

    def status(self) -> ProviderStatus: ...


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Turns text into audio, and optionally plays it."""

    name: str

    def synthesize(self, text: str) -> bytes:
        """WAV bytes for one sentence."""

    def speak(self, sentences: Iterable[str], stop_flag: Callable[[], bool]) -> float:
        """Synthesise and play, stopping early when `stop_flag()` goes true. Returns ms."""

    def status(self) -> ProviderStatus: ...


@dataclass(slots=True)
class ProviderSet:
    """The three providers a session is running on, resolved once."""

    stt: Transcriber
    llm: LanguageModel
    tts: SpeechSynthesizer
    warnings: list[str] = field(default_factory=list)

    def statuses(self) -> list[ProviderStatus]:
        return [self.stt.status(), self.llm.status(), self.tts.status()]
