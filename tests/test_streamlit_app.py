"""
The pure logic in the Streamlit entry point: which I/O mode the app picks, and
how it presents the personas.

Rendering is not exercised — that needs a live script run context — so these
cover the decisions made before anything is drawn. Nothing here touches an
audio device: `audio_available` is replaced.
"""

from __future__ import annotations

import pytest

streamlit_app = pytest.importorskip(
    "streamlit_app", reason="streamlit is not installed in this environment"
)

from src.persona_loader import PERSONA_DISPLAY_NAMES


@pytest.fixture
def no_microphone(monkeypatch):
    import src.audio_devices as audio_devices

    monkeypatch.setattr(audio_devices, "audio_available", lambda: False)


@pytest.fixture
def with_microphone(monkeypatch):
    import src.audio_devices as audio_devices

    monkeypatch.setattr(audio_devices, "audio_available", lambda: True)


class TestTextModeSelection:
    def test_voice_io_text_forces_text_mode_even_with_a_microphone(
        self, monkeypatch, with_microphone
    ) -> None:
        monkeypatch.setenv("VOICE_IO", "text")
        assert streamlit_app.text_mode_enabled() is True

    def test_voice_io_voice_forces_voice_mode_even_without_one(
        self, monkeypatch, no_microphone
    ) -> None:
        # An explicit choice must win, so a misdetected device can be overridden.
        monkeypatch.setenv("VOICE_IO", "voice")
        assert streamlit_app.text_mode_enabled() is False

    def test_the_setting_is_case_insensitive(self, monkeypatch, with_microphone) -> None:
        monkeypatch.setenv("VOICE_IO", "TEXT")
        assert streamlit_app.text_mode_enabled() is True

    @pytest.mark.parametrize("value", ["auto", "", "nonsense"])
    def test_otherwise_the_absence_of_a_microphone_decides(
        self, monkeypatch, no_microphone, value
    ) -> None:
        monkeypatch.setenv("VOICE_IO", value)
        assert streamlit_app.text_mode_enabled() is True

    def test_a_machine_with_a_microphone_gets_voice_mode(
        self, monkeypatch, with_microphone
    ) -> None:
        monkeypatch.delenv("VOICE_IO", raising=False)
        assert streamlit_app.text_mode_enabled() is False

    def test_an_unset_variable_behaves_like_auto(self, monkeypatch, no_microphone) -> None:
        monkeypatch.delenv("VOICE_IO", raising=False)
        assert streamlit_app.text_mode_enabled() is True


class TestPersonaPresentation:
    def test_personas_are_listed_under_their_display_names(self) -> None:
        names = [name for name, _ in streamlit_app.load_personas()]
        assert set(names) == set(PERSONA_DISPLAY_NAMES.values())

    def test_each_entry_carries_the_full_persona(self) -> None:
        for _, data in streamlit_app.load_personas():
            assert data["system_prompt"]
            assert data["scenario"]

    def test_display_names_are_unique(self) -> None:
        # They key the selectbox; a duplicate would make one scenario
        # unreachable from the UI.
        names = [name for name, _ in streamlit_app.load_personas()]
        assert len(names) == len(set(names))
