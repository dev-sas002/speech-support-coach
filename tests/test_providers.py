"""
The provider seam: how a stage is chosen, and what happens when it cannot be.

Nothing here builds a real client. The registries are generic, so these tests
register their own fakes into throwaway registries where the resolution rules
can be checked in isolation, and only touch the shipped registrations to assert
that the always-available stand-ins really are always available.
"""

from __future__ import annotations

import io
import wave

import pytest

from src.providers import (
    ProviderNotRegistered,
    available_providers,
    create_llm,
    create_provider_set,
    create_stt,
    create_tts,
)
from src.providers.registry import _Registry


@pytest.fixture
def registry(monkeypatch) -> _Registry:
    reg: _Registry = _Registry("widget", "WIDGET_PROVIDER")
    monkeypatch.delenv("WIDGET_PROVIDER", raising=False)

    @reg.register("hosted", available=lambda: False, priority=10)
    def _hosted() -> str:
        return "hosted-instance"

    @reg.register("local", priority=90)
    def _local() -> str:
        return "local-instance"

    return reg


class TestResolution:
    def test_an_explicit_name_wins_over_everything(self, registry, monkeypatch) -> None:
        monkeypatch.setenv("WIDGET_PROVIDER", "local")
        assert registry.resolve("hosted") == "hosted"

    def test_the_environment_selects_when_no_argument_is_given(self, registry, monkeypatch) -> None:
        monkeypatch.setenv("WIDGET_PROVIDER", "hosted")
        assert registry.resolve() == "hosted"

    def test_the_name_is_case_insensitive(self, registry, monkeypatch) -> None:
        monkeypatch.setenv("WIDGET_PROVIDER", "  LOCAL  ")
        assert registry.resolve() == "local"

    def test_auto_skips_a_provider_that_cannot_run_here(self, registry) -> None:
        # The hosted entry has the better priority but reports itself
        # unavailable, which is exactly the no-API-key case.
        assert registry.resolve() == "local"

    def test_auto_prefers_the_better_priority_once_it_is_usable(self, registry) -> None:
        registry._entries["hosted"].available = lambda: True
        assert registry.resolve() == "hosted"

    def test_an_explicit_unavailable_provider_is_still_built(self, registry) -> None:
        # An explicit choice must surface the provider's own failure rather
        # than being silently swapped for something else.
        assert registry.create("hosted") == "hosted-instance"

    def test_an_unknown_name_names_what_is_registered(self, registry) -> None:
        with pytest.raises(ProviderNotRegistered, match="hosted, local"):
            registry.resolve("wishful")

    def test_an_empty_environment_variable_means_auto(self, registry, monkeypatch) -> None:
        monkeypatch.setenv("WIDGET_PROVIDER", "")
        assert registry.resolve() == "local"


class TestShippedRegistrations:
    def test_every_stage_registers_a_stand_in(self) -> None:
        names = available_providers()
        assert "scripted" in names["llm"]
        assert "unavailable" in names["stt"]
        assert "silent" in names["tts"]

    def test_with_no_key_the_model_falls_back_to_the_script(self) -> None:
        # conftest strips GROQ_API_KEY for every test, so this is the
        # no-credentials path a fresh clone hits.
        assert create_llm().name == "scripted"

    def test_with_no_key_transcription_says_so_instead_of_guessing(self) -> None:
        transcriber = create_stt()
        assert transcriber.name == "unavailable"
        transcript = transcriber.transcribe(b"anything")
        assert "unavailable" in transcript.text
        assert transcript.latency_ms == 0.0

    def test_an_explicit_groq_selection_without_a_key_explains_itself(self) -> None:
        with pytest.raises(Exception, match="GROQ_API_KEY"):
            create_llm("groq")

    def test_the_scripted_model_answers_in_character(self) -> None:
        model = create_llm("scripted")
        reply = model.complete(
            [{"role": "system", "content": "The customer lost their debit card."}]
        )
        assert "card" in reply.text.lower()
        assert reply.latency_ms >= 0
        assert reply.total_tokens

    def test_the_scripted_model_streams(self) -> None:
        model = create_llm("scripted")
        pieces = list(model.stream([{"role": "system", "content": "lost card"}]))
        assert len(pieces) > 1
        assert "".join(pieces).strip()


class TestSilentSynthesis:
    def test_it_produces_a_wav_that_actually_parses(self) -> None:
        # The whole point of silence over a no-op is that every downstream
        # consumer stays on the real code path.
        wav_bytes = create_tts("silent").synthesize("Two short sentences here.")
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getnframes() > 0

    def test_longer_text_produces_longer_audio(self) -> None:
        tts = create_tts("silent")
        assert len(tts.synthesize("a" * 200)) > len(tts.synthesize("a" * 20))

    def test_speaking_honours_the_stop_flag(self) -> None:
        spoken: list[str] = []
        tts = create_tts("silent")

        def sentences():
            for text in ["one.", "two.", "three."]:
                spoken.append(text)
                yield text

        tts.speak(sentences(), lambda: len(spoken) >= 2)
        assert spoken == ["one.", "two."]

    def test_it_reports_itself_as_not_ready(self) -> None:
        # The UI shows this so nobody thinks the silence is a bug.
        assert create_tts("silent").status().ready is False


class TestProviderSet:
    def test_all_three_stages_resolve_without_any_credentials(self) -> None:
        providers = create_provider_set()
        assert [s.resolved for s in providers.statuses()] == [
            "unavailable",
            "scripted",
            "silent",
        ]
        assert providers.warnings == []

    def test_a_stage_that_cannot_start_degrades_with_a_warning(self, monkeypatch) -> None:
        # A misconfigured stage must not take the session down: the rest of the
        # pipeline is still worth running, and the UI needs to say what broke.
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        providers = create_provider_set()

        assert providers.llm.name == "scripted"
        assert any("groq" in warning for warning in providers.warnings)

    def test_an_unknown_provider_name_degrades_rather_than_raising(self, monkeypatch) -> None:
        monkeypatch.setenv("TTS_PROVIDER", "not-a-real-provider")
        providers = create_provider_set()

        assert providers.tts.name == "silent"
        assert providers.warnings
