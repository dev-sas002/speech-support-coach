"""
Lazy access to the audio device libraries.

`sounddevice` and `simpleaudio` bind to PortAudio and ALSA at import time, so
importing them at module scope makes every module that touches them
unimportable on a machine with no audio stack — a container, a CI runner, a
headless server. That is how the test suite failed: collecting
`tests/test_voice_pipeline.py` raised `OSError: PortAudio library not found`
before a single test ran, and took the other suites down with it.

Microphone capture and speaker playback genuinely need those libraries.
Importing a conversation module does not. This defers the import to the point
of use and raises a message that says what is actually missing.
"""

from __future__ import annotations

from typing import Any


class AudioUnavailable(RuntimeError):
    """Raised when audio hardware or its driver libraries are not present."""


def get_sounddevice() -> Any:
    """The `sounddevice` module, or a clear error explaining what is missing."""
    try:
        import sounddevice

        return sounddevice
    except (ImportError, OSError) as exc:
        raise AudioUnavailable(
            "Microphone capture needs the PortAudio library, which is not "
            "available here (containers and CI runners normally have no audio "
            "stack). Use text mode, or install libportaudio2."
        ) from exc


def get_simpleaudio() -> Any:
    """The `simpleaudio` module, or a clear error explaining what is missing."""
    try:
        import simpleaudio

        return simpleaudio
    except (ImportError, OSError) as exc:
        raise AudioUnavailable(
            "Speaker playback needs the ALSA libraries, which are not "
            "available here. Synthesised audio can still be written to a file."
        ) from exc


def get_webrtcvad() -> Any:
    """The `webrtcvad` module, or a clear error explaining what is missing."""
    try:
        import webrtcvad

        return webrtcvad
    except (ImportError, OSError) as exc:
        raise AudioUnavailable(
            "Voice activity detection needs the webrtcvad extension, which is "
            "not installed here. It is only required for microphone capture; "
            "install the audio extras with "
            "`pip install -r requirements/audio.txt`."
        ) from exc


def audio_available() -> bool:
    """Whether microphone capture is possible in this environment."""
    try:
        get_sounddevice()
        return True
    except AudioUnavailable:
        return False
