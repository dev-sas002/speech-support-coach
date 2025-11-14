"""
Speech synthesis and playback.

Nothing here opens a speaker: the `simpleaudio` proxy is replaced with a
recorder, and the Kokoro pipeline is replaced with a callable returning fixed
audio. No test touches the network or a real audio device.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from src import tts_module
from src.tts_module import KokoroTTSClient, PlaybackController, _read_wav_params


def _wav(pcm: bytes = b"\x01\x02" * 400, rate: int = 24000) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return bio.getvalue()


class _FakePlayObject:
    def __init__(self, frames: bytes, playing_for: int = 0) -> None:
        self.frames = frames
        self.stopped = False
        self.waited = False
        self._remaining = playing_for

    def wait_done(self) -> None:
        self.waited = True

    def is_playing(self) -> bool:
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True

    def stop(self) -> None:
        self.stopped = True


class _FakeSimpleAudio:
    def __init__(self, playing_for: int = 0) -> None:
        self.played: list[_FakePlayObject] = []
        self._playing_for = playing_for

    def play_buffer(self, frames, nchannels, sampwidth, framerate):
        obj = _FakePlayObject(frames, self._playing_for)
        self.played.append(obj)
        return obj


@pytest.fixture
def fake_sa(monkeypatch):
    """Replace the lazy simpleaudio proxy so no speaker is opened."""
    fake = _FakeSimpleAudio()
    monkeypatch.setattr(tts_module, "sa", fake)
    return fake


# --- the WAV reader --------------------------------------------------------


class TestReadWavParams:
    def test_it_reads_the_buffer_it_was_given(self) -> None:
        # Regression: this opened an *empty* BytesIO instead of one wrapping
        # `wav_bytes`, so every call raised EOFError and playback never worked.
        pcm = b"\x10\x20" * 300
        params, frames = _read_wav_params(_wav(pcm))

        assert frames == pcm
        assert params.nchannels == 1
        assert params.sampwidth == 2
        assert params.framerate == 24000

    def test_a_buffer_that_is_not_a_wav_is_rejected(self) -> None:
        # `wave` raises EOFError on a truncated header and wave.Error on a
        # malformed one; both mean the same thing here, and naming them is
        # better than accepting any exception at all.
        with pytest.raises((EOFError, wave.Error)):
            _read_wav_params(b"definitely not a wav file")


# --- playback --------------------------------------------------------------


class TestPlaybackController:
    def test_play_wav_sends_the_decoded_frames_to_the_device(self, fake_sa) -> None:
        pcm = b"\x05\x06" * 200
        PlaybackController().play_wav(_wav(pcm))

        assert len(fake_sa.played) == 1
        assert fake_sa.played[0].frames == pcm
        assert fake_sa.played[0].waited is True

    def test_play_wav_releases_the_handle_when_it_finishes(self, fake_sa) -> None:
        pc = PlaybackController()
        pc.play_wav(_wav())
        assert pc._current is None

    def test_play_wav_releases_the_handle_when_playback_fails(self, monkeypatch) -> None:
        # A raising wait_done used to leave a dead PlayObject referenced here,
        # and the next stop() acted on it.
        class _Boom(_FakeSimpleAudio):
            def play_buffer(self, *args):
                obj = super().play_buffer(*args)
                obj.wait_done = lambda: (_ for _ in ()).throw(OSError("device gone"))
                return obj

        monkeypatch.setattr(tts_module, "sa", _Boom())
        pc = PlaybackController()
        with pytest.raises(OSError):
            pc.play_wav(_wav())
        assert pc._current is None

    def test_stop_is_safe_when_nothing_is_playing(self) -> None:
        pc = PlaybackController()
        pc.stop()  # cleanup runs on a controller that never played
        assert pc._current is None

    def test_stop_halts_the_current_playback(self, fake_sa) -> None:
        fake_sa._playing_for = 100
        pc = PlaybackController()
        pc.play_wav_interruptible(_wav(), stop_flag=lambda: False)
        # The fake stops playing on its own; stop() after the fact is a no-op.
        pc.stop()
        assert pc._current is None

    def test_interruptible_playback_stops_on_the_flag(self, monkeypatch) -> None:
        monkeypatch.setattr(tts_module.time, "sleep", lambda _s: None)
        fake = _FakeSimpleAudio(playing_for=100)
        monkeypatch.setattr(tts_module, "sa", fake)

        pc = PlaybackController()
        pc.play_wav_interruptible(_wav(), stop_flag=lambda: True)

        assert fake.played[0].stopped is True
        assert pc._current is None

    def test_interruptible_playback_runs_to_the_end_when_not_interrupted(self, monkeypatch) -> None:
        monkeypatch.setattr(tts_module.time, "sleep", lambda _s: None)
        fake = _FakeSimpleAudio(playing_for=3)
        monkeypatch.setattr(tts_module, "sa", fake)

        PlaybackController().play_wav_interruptible(_wav(), stop_flag=lambda: False)
        assert fake.played[0].stopped is False


# --- synthesis -------------------------------------------------------------


def _client_without_kokoro(monkeypatch, *, allow_fallback: bool = False) -> KokoroTTSClient:
    monkeypatch.setenv("ALLOW_FALLBACK_TTS", "1" if allow_fallback else "0")
    client = KokoroTTSClient()
    client.use_kokoro = False
    client.pipeline = None
    return client


class TestSynthesis:
    def test_a_missing_kokoro_does_not_break_the_import_or_the_constructor(
        self, monkeypatch
    ) -> None:
        # The kokoro import used to sit at module scope, which made the
        # except-branch in the constructor unreachable: with kokoro absent the
        # import failed first and took the whole module down.
        client = _client_without_kokoro(monkeypatch)
        assert client.use_kokoro is False

    def test_kokoro_audio_is_framed_as_a_readable_wav(self, monkeypatch) -> None:
        client = _client_without_kokoro(monkeypatch)
        samples = np.linspace(-1.0, 1.0, 240, dtype=np.float32)

        client.use_kokoro = True
        client.pipeline = lambda text, voice, speed: [("gs", "ps", samples)]

        wav = client.synthesize_sentence("hello")
        with wave.open(io.BytesIO(wav), "rb") as wf:
            assert wf.getframerate() == client.sample_rate
            assert wf.getnframes() == len(samples)

    def test_multiple_kokoro_chunks_are_concatenated(self, monkeypatch) -> None:
        client = _client_without_kokoro(monkeypatch)
        chunk = np.zeros(100, dtype=np.float32)

        client.use_kokoro = True
        client.pipeline = lambda text, voice, speed: [
            ("g", "p", chunk),
            ("g", "p", chunk),
            ("g", "p", chunk),
        ]

        with wave.open(io.BytesIO(client.synthesize_sentence("x")), "rb") as wf:
            assert wf.getnframes() == 300

    def test_no_kokoro_and_no_fallback_is_an_explicit_failure(self, monkeypatch) -> None:
        client = _client_without_kokoro(monkeypatch)
        with pytest.raises(RuntimeError, match="not configured and fallback disabled"):
            client.synthesize_sentence("hello")

    def test_a_kokoro_failure_propagates_when_fallback_is_off(self, monkeypatch) -> None:
        client = _client_without_kokoro(monkeypatch)
        client.use_kokoro = True

        def _boom(text, voice, speed):
            raise ValueError("model exploded")

        client.pipeline = _boom
        with pytest.raises(ValueError, match="model exploded"):
            client.synthesize_sentence("hello")

    def test_empty_kokoro_output_falls_through_rather_than_returning_silence(
        self, monkeypatch
    ) -> None:
        client = _client_without_kokoro(monkeypatch)
        client.use_kokoro = True
        client.pipeline = lambda text, voice, speed: []

        with pytest.raises(RuntimeError):
            client.synthesize_sentence("hello")

    def test_the_fallback_is_bounded_and_survives_a_missing_say_binary(self, monkeypatch) -> None:
        # `say` exists only on macOS and the call had no timeout, so on Linux
        # the safety net itself raised FileNotFoundError.
        client = _client_without_kokoro(monkeypatch, allow_fallback=True)

        import subprocess

        recorded: dict = {}

        def _fake_run(cmd, **kwargs):
            recorded.update(kwargs)
            raise FileNotFoundError("say: not found")

        monkeypatch.setattr(subprocess, "run", _fake_run)

        with pytest.raises(RuntimeError, match="not configured and fallback disabled"):
            client.synthesize_sentence("hello")
        assert recorded["timeout"] == tts_module.FALLBACK_TTS_TIMEOUT_S

    def test_the_fallback_produces_playable_audio_when_say_works(self, monkeypatch) -> None:
        client = _client_without_kokoro(monkeypatch, allow_fallback=True)

        import subprocess

        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: None)

        wav = client.synthesize_sentence("hello")
        params, frames = _read_wav_params(wav)
        assert params.framerate == client.sample_rate
        assert len(frames) > 0


# --- speaking a turn -------------------------------------------------------


class TestSpeakSentences:
    def _client(self, monkeypatch) -> KokoroTTSClient:
        client = _client_without_kokoro(monkeypatch)
        client.use_kokoro = True
        client.pipeline = lambda text, voice, speed: [("g", "p", np.zeros(48, dtype=np.float32))]
        return client

    def test_every_sentence_reaches_the_speaker(self, monkeypatch, fake_sa) -> None:
        client = self._client(monkeypatch)
        elapsed = client.speak_sentences(["one.", "two.", "three."], lambda: False)

        assert len(fake_sa.played) == 3
        assert elapsed >= 0

    def test_a_stop_before_the_first_sentence_speaks_nothing(self, monkeypatch, fake_sa) -> None:
        client = self._client(monkeypatch)
        client.speak_sentences(["one.", "two."], lambda: True)
        assert fake_sa.played == []

    def test_a_stop_partway_through_ends_the_turn(self, monkeypatch, fake_sa) -> None:
        client = self._client(monkeypatch)
        calls = {"n": 0}

        def stop_after_first() -> bool:
            calls["n"] += 1
            return calls["n"] > 2

        client.speak_sentences(["one.", "two.", "three."], stop_after_first)
        assert len(fake_sa.played) < 3

    def test_the_sentence_source_is_consumed_lazily(self, monkeypatch, fake_sa) -> None:
        # The caller passes a generator that is still being filled by the LLM
        # stream; barge-in must stop pulling from it, not drain it first.
        client = self._client(monkeypatch)
        pulled: list[str] = []

        def sentences():
            for s in ["one.", "two.", "three."]:
                pulled.append(s)
                yield s

        calls = {"n": 0}

        def stop_after_one() -> bool:
            calls["n"] += 1
            return calls["n"] > 2

        client.speak_sentences(sentences(), stop_after_one)
        assert len(pulled) < 3
