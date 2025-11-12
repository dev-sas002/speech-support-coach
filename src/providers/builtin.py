"""
The providers that ship with the project.

Each is a thin adapter over a client that already existed: `ASRClient`,
`LLMClient` and `KokoroTTSClient` keep the vendor-specific details, and these
classes expose them through the protocols in `base.py`. Adding a provider means
adding a class here (or in your own module) and one `@register` line — nothing
above the seam changes.

Every stage has a stand-in that always works, so the app runs with no
credentials and no audio hardware. The stand-ins are deliberately honest: the
recogniser says it cannot hear rather than inventing words, and the synthesiser
produces silence rather than pretending a speaker exists.
"""

from __future__ import annotations

import importlib.util
import io
import time
import wave
from collections.abc import Callable, Iterable, Iterator

from ..local_backend import LOCAL_MODEL_NAME, LocalGroqClient, usable_api_key
from .base import (
    LLMReply,
    Message,
    ProviderStatus,
    ProviderUnavailable,
    Transcript,
)
from .registry import llm_registry, stt_registry, tts_registry


def _module_present(name: str) -> bool:
    """Whether a module could be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _have_key() -> bool:
    return usable_api_key("GROQ_API_KEY") is not None


def _silence_wav(seconds: float, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(sample_rate * max(seconds, 0.0)))
    return buffer.getvalue()


# --- Language models -------------------------------------------------------


class _LLMClientAdapter:
    """Shared body for the providers backed by `LLMClient`."""

    name = "llm"
    _detail = ""

    def __init__(self, client) -> None:
        self._client = client

    @property
    def model(self) -> str:
        return self._client.config.model

    def complete(self, messages: list[Message]) -> LLMReply:
        text, latency_ms, usage = self._client.complete(messages)
        return LLMReply(text=text, latency_ms=latency_ms, usage=usage)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        tokens, _ = self._client.stream_chat(messages)
        return tokens

    @property
    def last_first_token_ms(self) -> float | None:
        """Time to first token of the most recent `stream()`, once it has run."""
        return self._client.last_first_token_ms

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            kind="llm",
            requested=llm_registry.requested(),
            resolved=self.name,
            ready=True,
            detail=self._detail.format(model=self.model),
        )


@llm_registry.register("groq", available=_have_key, priority=10)
def _groq_llm() -> _LLMClientAdapter:
    from ..llm_module import LLMClient

    if not _have_key():
        raise ProviderUnavailable(
            "GROQ_API_KEY is not set (or is still the placeholder from "
            ".env.example), so the hosted model cannot be reached."
        )
    adapter = _LLMClientAdapter(LLMClient())
    adapter.name = "groq"
    adapter._detail = "Hosted model {model}"
    return adapter


@llm_registry.register("scripted", priority=90)
def _scripted_llm() -> _LLMClientAdapter:
    from ..llm_module import LLMClient

    adapter = _LLMClientAdapter(LLMClient(model=LOCAL_MODEL_NAME, client=LocalGroqClient()))
    adapter.name = "scripted"
    adapter._detail = "Fixed support-agent script, no network"
    return adapter


# --- Speech to text --------------------------------------------------------


@stt_registry.register("groq", available=_have_key, priority=10)
def _groq_stt() -> GroqTranscriber:
    if not _have_key():
        raise ProviderUnavailable(
            "GROQ_API_KEY is not set, so hosted transcription is unavailable."
        )
    return GroqTranscriber()


class GroqTranscriber:
    name = "groq"

    def __init__(self) -> None:
        from ..asr_module import ASRClient

        self._client = ASRClient()

    def transcribe(self, wav_bytes: bytes) -> Transcript:
        text, latency_ms = self._client.transcribe_wav_bytes(wav_bytes)
        frames = max(len(wav_bytes) - 44, 0) / 2
        return Transcript(
            text=text,
            latency_ms=latency_ms,
            audio_seconds=frames / self._client.sample_rate,
        )

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            kind="stt",
            requested=stt_registry.requested(),
            resolved=self.name,
            ready=True,
            detail=f"Hosted {self._client.config.model}",
        )


@stt_registry.register("unavailable", priority=90)
def _unavailable_stt() -> UnavailableTranscriber:
    return UnavailableTranscriber()


class UnavailableTranscriber:
    """
    No local recogniser.

    Returning invented words would be worse than returning nothing: the rest of
    the pipeline cannot tell a guess from a transcript, and a bank-support
    trainer that silently rewrites what the caller said is misleading.
    """

    name = "unavailable"
    MESSAGE = "[speech recognition unavailable — set GROQ_API_KEY, or type your message]"

    def transcribe(self, wav_bytes: bytes) -> Transcript:
        return Transcript(text=self.MESSAGE, latency_ms=0.0, audio_seconds=0.0)

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            kind="stt",
            requested=stt_registry.requested(),
            resolved=self.name,
            ready=False,
            detail="No recogniser here — type instead of speaking",
        )


# --- Speech synthesis ------------------------------------------------------


@tts_registry.register("kokoro", available=lambda: _module_present("kokoro"), priority=10)
def _kokoro_tts() -> KokoroSynthesizer:
    return KokoroSynthesizer()


class KokoroSynthesizer:
    name = "kokoro"

    def __init__(self) -> None:
        from ..tts_module import KokoroTTSClient

        self._client = KokoroTTSClient()
        if not self._client.use_kokoro and not self._client.allow_fallback:
            raise ProviderUnavailable(
                "Kokoro could not be initialised and the OS fallback is off "
                "(set ALLOW_FALLBACK_TTS=1 to permit it)."
            )

    def synthesize(self, text: str) -> bytes:
        return self._client.synthesize_sentence(text)

    def speak(self, sentences: Iterable[str], stop_flag: Callable[[], bool]) -> float:
        return self._client.speak_sentences(sentences, stop_flag)

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            kind="tts",
            requested=tts_registry.requested(),
            resolved=self.name,
            ready=True,
            detail=f"Kokoro voice {self._client.voice} at {self._client.sample_rate} Hz",
        )


@tts_registry.register("silent", priority=90)
def _silent_tts() -> SilentSynthesizer:
    return SilentSynthesizer()


class SilentSynthesizer:
    """
    Valid WAV output, no speaker.

    Containers and CI runners have no audio device. Producing well-formed
    silence keeps every downstream consumer — WAV parsing, file export, the
    duration arithmetic — on the same code path as real audio, which is the
    point: the pipeline is exercised, only the loudspeaker is missing.
    """

    name = "silent"
    #: Roughly conversational pace, used to give the silence a realistic length.
    CHARS_PER_SECOND = 15.0

    def synthesize(self, text: str) -> bytes:
        return _silence_wav(len(text) / self.CHARS_PER_SECOND)

    def speak(self, sentences: Iterable[str], stop_flag: Callable[[], bool]) -> float:
        started = time.perf_counter()
        for sentence in sentences:
            if stop_flag():
                break
            self.synthesize(sentence)
        return (time.perf_counter() - started) * 1000

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            kind="tts",
            requested=tts_registry.requested(),
            resolved=self.name,
            ready=False,
            detail="No audio device — synthesis produces silence",
        )
