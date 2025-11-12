import contextlib
import io
import os
import time
import wave
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

#: Upper bound on the macOS `say` fallback, so a wedged process cannot block a turn.
FALLBACK_TTS_TIMEOUT_S = float(os.getenv("FALLBACK_TTS_TIMEOUT_S", "30"))


class _LazySimpleAudio:
    """Stands in for `simpleaudio` until playback is actually requested."""

    def __getattr__(self, name):
        from .audio_devices import get_simpleaudio

        return getattr(get_simpleaudio(), name)


sa = _LazySimpleAudio()


def _read_wav_params(wav_bytes: bytes):
    """Return (params, frames) for a WAV byte buffer."""
    # This opened an *empty* BytesIO rather than one wrapping `wav_bytes`, so
    # every call raised EOFError inside `wave.open` and playback never worked.
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
        return params, frames


class PlaybackController:
    """
    Handle playback of WAV audio, including an interruptible variant for barge-in.
    """

    def __init__(self) -> None:
        self._current: Any | None = None  # simpleaudio.PlayObject

    def stop(self) -> None:
        if self._current is not None:
            # The device may already be gone; releasing the handle is the point.
            with contextlib.suppress(Exception):
                self._current.stop()
            self._current = None

    def play_wav(self, wav_bytes: bytes) -> None:
        params, frames = _read_wav_params(wav_bytes)
        play = sa.play_buffer(frames, params.nchannels, params.sampwidth, params.framerate)
        self._current = play
        try:
            play.wait_done()
        finally:
            # Without the finally, a failed or interrupted playback left a
            # dead PlayObject referenced here, and the next stop() acted on it.
            self._current = None

    def play_wav_interruptible(self, wav_bytes: bytes, stop_flag: Callable[[], bool]) -> None:
        params, frames = _read_wav_params(wav_bytes)
        play = sa.play_buffer(frames, params.nchannels, params.sampwidth, params.framerate)
        self._current = play
        try:
            while play.is_playing():
                if stop_flag():
                    with contextlib.suppress(Exception):
                        play.stop()
                    break
                time.sleep(0.02)
        finally:
            self._current = None


class KokoroTTSClient:
    """
    Text-to-speech client backed by Kokoro, with optional macOS `say` fallback.
    """

    def __init__(self) -> None:
        self.voice = os.getenv("KOKORO_VOICE", "af_sky")
        self.lang_code = os.getenv("KOKORO_LANG_CODE", "a")  # 'a' = American English
        self.sample_rate = 24000  # Kokoro outputs at 24kHz
        self.allow_fallback = os.getenv("ALLOW_FALLBACK_TTS", "0") == "1"
        self.playback = PlaybackController()

        try:
            # `from kokoro import KPipeline` at module scope pulled in torch and
            # transformers just to import this module, and made the except-branch
            # below unreachable: with kokoro absent the import failed first and
            # took down every module that touches the conversation pipeline.
            from kokoro import KPipeline

            self.pipeline = KPipeline(lang_code=self.lang_code)
            self.use_kokoro = True
        except Exception as e:
            print(f"Warning: Failed to initialize Kokoro pipeline: {e}")
            self.pipeline = None
            self.use_kokoro = False

    def _synthesize_with_kokoro(self, text: str) -> bytes | None:
        if not (self.use_kokoro and self.pipeline):
            return None
        generator = self.pipeline(text, voice=self.voice, speed=1.0)
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)
        if not audio_chunks:
            return None

        full_audio = np.concatenate(audio_chunks)
        audio_int16 = (full_audio * 32767).astype(np.int16)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        wav_io.seek(0)
        return wav_io.read()

    def _fallback_tts(self, text: str) -> bytes | None:
        if not self.allow_fallback:
            return None
        import subprocess

        tmp = text.replace("\n", " ")
        # `say` exists only on macOS and had no timeout: on Linux this raised
        # FileNotFoundError out of the fallback that was meant to be the safety
        # net, and a wedged process blocked the turn indefinitely.
        try:
            subprocess.run(
                ["say", tmp],
                timeout=FALLBACK_TTS_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"Fallback TTS unavailable: {exc}")
            return None
        wav = io.BytesIO()
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"\x00" * int(self.sample_rate * 0.2) * 2)
        return wav.getvalue()

    def synthesize_sentence(self, text: str) -> bytes:
        """
        Convert a sentence of text into WAV bytes.

        Preference order:
        1. Kokoro pipeline (when available)
        2. Optional OS-level fallback (macOS `say`) with dummy audio padding
        """
        try:
            wav = self._synthesize_with_kokoro(text)
            if wav is not None:
                return wav
        except Exception as e:
            print(f"Kokoro TTS error: {e}")
            if not self.allow_fallback:
                raise

        wav = self._fallback_tts(text)
        if wav is not None:
            return wav

        raise RuntimeError("Kokoro TTS not configured and fallback disabled")

    def speak_sentences(self, sentences: Iterable[str], stop_flag: Callable[[], bool]) -> float:
        """
        Speak a sequence of sentences, honoring a stop flag for barge-in.
        """
        t0 = time.perf_counter()
        for s in sentences:
            if stop_flag():
                break
            wav = self.synthesize_sentence(s)
            if stop_flag():
                break
            self.playback.play_wav_interruptible(wav, stop_flag)
        return (time.perf_counter() - t0) * 1000
