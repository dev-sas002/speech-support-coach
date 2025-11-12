"""
The provider registry: name -> factory, resolved from configuration.

Three registries rather than one, because the three stages are chosen
independently — a hosted recogniser with a local synthesiser is a normal
combination, not a mistake.

Resolution order for each stage is: an explicit argument, then the environment
variable (`STT_PROVIDER`, `LLM_PROVIDER`, `TTS_PROVIDER`), then `auto`. `auto`
asks each registered factory's availability probe and picks the first one that
can actually run, in registration order, so the app starts with no credentials
and degrades to whatever is possible here rather than raising.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .base import (
    LanguageModel,
    ProviderNotRegistered,
    ProviderSet,
    SpeechSynthesizer,
    Transcriber,
)

T = TypeVar("T")

AUTO = "auto"


@dataclass(slots=True)
class _Entry(Generic[T]):
    name: str
    factory: Callable[[], T]
    #: Whether this provider can run in the current environment. `auto` skips
    #: those that answer False; an explicit selection still gets built, so a
    #: misconfiguration produces the provider's own error rather than silence.
    available: Callable[[], bool]
    #: Lower sorts first when `auto` chooses. Hosted providers are preferred
    #: over local stand-ins when both can run.
    priority: int


class _Registry(Generic[T]):
    def __init__(self, kind: str, env_var: str) -> None:
        self.kind = kind
        self.env_var = env_var
        self._entries: dict[str, _Entry[T]] = {}

    def register(
        self,
        name: str,
        *,
        available: Callable[[], bool] = lambda: True,
        priority: int = 50,
    ) -> Callable[[Callable[[], T]], Callable[[], T]]:
        def decorator(factory: Callable[[], T]) -> Callable[[], T]:
            self._entries[name] = _Entry(
                name=name, factory=factory, available=available, priority=priority
            )
            return factory

        return decorator

    def names(self) -> list[str]:
        return sorted(self._entries)

    def requested(self, name: str | None = None) -> str:
        if name:
            return name.strip().lower()
        return (os.getenv(self.env_var) or AUTO).strip().lower() or AUTO

    def resolve(self, name: str | None = None) -> str:
        """The concrete provider name this configuration selects."""
        wanted = self.requested(name)
        if wanted != AUTO:
            if wanted not in self._entries:
                raise ProviderNotRegistered(
                    f"No {self.kind} provider named '{wanted}'. "
                    f"Registered: {', '.join(self.names()) or 'none'}."
                )
            return wanted

        for entry in sorted(self._entries.values(), key=lambda e: (e.priority, e.name)):
            if entry.available():
                return entry.name
        raise ProviderNotRegistered(
            f"No {self.kind} provider is usable here and none is registered as a fallback."
        )

    def create(self, name: str | None = None) -> T:
        return self._entries[self.resolve(name)].factory()


stt_registry: _Registry[Transcriber] = _Registry("speech-to-text", "STT_PROVIDER")
llm_registry: _Registry[LanguageModel] = _Registry("language model", "LLM_PROVIDER")
tts_registry: _Registry[SpeechSynthesizer] = _Registry("text-to-speech", "TTS_PROVIDER")


def create_stt(name: str | None = None) -> Transcriber:
    _ensure_builtins()
    return stt_registry.create(name)


def create_llm(name: str | None = None) -> LanguageModel:
    _ensure_builtins()
    return llm_registry.create(name)


def create_tts(name: str | None = None) -> SpeechSynthesizer:
    _ensure_builtins()
    return tts_registry.create(name)


def create_provider_set(
    stt: str | None = None,
    llm: str | None = None,
    tts: str | None = None,
) -> ProviderSet:
    """
    Resolve all three stages at once.

    A stage that cannot be built is not allowed to take the session down: the
    failure is recorded as a warning and that stage falls back to its
    always-available stand-in, so the parts of the pipeline that do work still
    run and the UI can say what is missing.
    """
    _ensure_builtins()
    warnings: list[str] = []

    def build(registry: _Registry, wanted: str | None, fallback: str):
        try:
            return registry.create(wanted)
        except Exception as exc:
            warnings.append(
                f"{registry.kind}: could not start "
                f"'{registry.requested(wanted)}' ({exc}); using '{fallback}'."
            )
            return registry.create(fallback)

    return ProviderSet(
        stt=build(stt_registry, stt, "unavailable"),
        llm=build(llm_registry, llm, "scripted"),
        tts=build(tts_registry, tts, "silent"),
        warnings=warnings,
    )


def available_providers() -> dict[str, list[str]]:
    """Every registered provider name, by stage. Used by the UI and the docs."""
    _ensure_builtins()
    return {
        "stt": stt_registry.names(),
        "llm": llm_registry.names(),
        "tts": tts_registry.names(),
    }


_builtins_loaded = False


def _ensure_builtins() -> None:
    """Import the built-in providers once, on first use."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    from . import builtin  # noqa: F401  (import for its registration side effects)
