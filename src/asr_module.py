import io
import os
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from groq import Groq

from .local_backend import LocalGroqClient, request_timeout_s, usable_api_key


class _LazySoundDevice:
    """
    Stands in for the `sounddevice` module until something actually uses it.

    Binding PortAudio at import time makes this module unimportable wherever
    there is no audio stack — a container, a CI runner — which took down test
    collection for the whole suite. Attribute access resolves the real module
    on first use, so `sd.InputStream(...)` still reads normally and tests can
    still replace `sd` wholesale.
    """

    def __getattr__(self, name):
        from .audio_devices import get_sounddevice

        return getattr(get_sounddevice(), name)


class _LazyWebrtcVad:
    """
    Stands in for `webrtcvad` until a detector is actually constructed.

    It is a C extension with no wheel for every interpreter, and it is needed
    only for microphone capture. Importing it at module scope made the whole
    conversation pipeline — state, personas, the model call — uninstallable
    anywhere the extension could not be built, which is every slim container.
    """

    def __getattr__(self, name):
        from .audio_devices import get_webrtcvad

        return getattr(get_webrtcvad(), name)


sd = _LazySoundDevice()
webrtcvad = _LazyWebrtcVad()


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """
    Convert raw 16-bit PCM mono audio into WAV container bytes.
    """
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    bio.seek(0)
    return bio.read()


@dataclass(slots=True)
class ASRConfig:
    """Configuration for the ASR client and streaming loop."""

    model: str
    sample_rate: int = 16000
    partial_interval_ms: int = 800
    min_speech_ms: int = 200
    max_silence_ms: int = 600
    #: Hard ceiling on one capture. The listen loop had no exit other than
    #: speech-then-silence, so a silent or stuck microphone blocked the caller
    #: forever and VoiceClient.request_stop() could never take effect.
    max_utterance_ms: int = 30000


class ASRClient:
    """
    Wrapper around Groq's audio transcription API.

    The underlying Groq client can be injected for testing via the `client`
    parameter, which keeps network calls out of unit tests.
    """

    def __init__(
        self,
        model: str | None = None,
        sample_rate: int = 16000,
        client: Groq | None = None,
    ) -> None:
        api_key = usable_api_key("GROQ_API_KEY")
        if client is not None:
            self.client = client
        elif api_key:
            self.client = Groq(api_key=api_key, timeout=request_timeout_s())
        else:
            # Offline there is no speech recogniser; the stand-in returns a
            # message saying so rather than inventing words and letting the
            # pipeline treat them as what the trainee said.
            self.client = LocalGroqClient()

        self.is_local = api_key is None and client is None
        model_name = model or os.getenv("GROQ_ASR_MODEL", "whisper-large-v3-turbo")
        self.config = ASRConfig(model=model_name, sample_rate=sample_rate)

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> tuple[str, float]:
        """
        Transcribe a complete WAV byte buffer and return (text, latency_ms).
        """
        t0 = time.perf_counter()
        bio = io.BytesIO(wav_bytes)
        bio.name = "audio.wav"
        try:
            resp = self.client.audio.transcriptions.create(
                model=self.config.model,
                file=bio,
            )
        except Exception as exc:
            raise RuntimeError("Failed to transcribe audio") from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        text = getattr(resp, "text", "")
        return text, latency_ms

    def streaming_listen(
        self,
        vad_stream: "VADStream",
        on_partial: Callable[[str], None] = lambda t: None,
        partial_interval_ms: int | None = None,
        min_speech_ms: int | None = None,
        max_silence_ms: int | None = None,
        max_utterance_ms: int | None = None,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> tuple[str, float, float]:
        """
        Listen on a VADStream until a segment of speech is detected and silence follows.

        Capture also ends when `max_utterance_ms` of wall clock has elapsed or
        `should_stop()` becomes true, so the caller is never blocked
        indefinitely by a silent or stuck microphone.

        Returns (final_text, asr_latency_ms, asr_audio_seconds).
        """
        cfg = self.config
        partial_interval = partial_interval_ms or cfg.partial_interval_ms
        min_speech = min_speech_ms or cfg.min_speech_ms
        max_silence = max_silence_ms or cfg.max_silence_ms
        max_utterance = max_utterance_ms or cfg.max_utterance_ms

        buf = bytearray()
        started = False
        voiced_ms = 0
        unvoiced_ms = 0
        capture_start = time.perf_counter()
        last_partial_time = capture_start

        while True:
            if should_stop():
                break
            if (time.perf_counter() - capture_start) * 1000 >= max_utterance:
                break
            data = vad_stream.stream.read(int(vad_stream.frame_bytes / 2))[0].tobytes()
            if len(data) < vad_stream.frame_bytes:
                continue
            is_speech = vad_stream.vad.is_speech(data, vad_stream.sample_rate)
            if is_speech:
                buf.extend(data)
                voiced_ms += 30
                unvoiced_ms = 0
                if not started and voiced_ms >= min_speech:
                    started = True
            else:
                if started:
                    unvoiced_ms += 30
                    if unvoiced_ms >= max_silence:
                        break
                else:
                    voiced_ms = 0

            if started:
                now = time.perf_counter()
                enough_audio = len(buf) > int(self.sample_rate * 0.5) * 2
                if (now - last_partial_time) * 1000 >= partial_interval and enough_audio:
                    try:
                        wav = pcm16_to_wav_bytes(bytes(buf), self.sample_rate)
                        text, _ = self.transcribe_wav_bytes(wav)
                        if text:
                            on_partial(text)
                    except Exception:
                        # Partial failures should not abort the full capture.
                        pass
                    last_partial_time = now

        wav = pcm16_to_wav_bytes(bytes(buf), self.sample_rate)
        final_text, final_ms = self.transcribe_wav_bytes(wav)
        asr_secs = len(buf) / 2 / self.sample_rate
        return final_text, final_ms, asr_secs


class VADStream:
    """
    Voice activity detector stream built on top of webrtcvad and sounddevice.
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, aggressiveness: int = 2):
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.stream: Any | None = None  # sounddevice.InputStream

    def start(self) -> None:
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16")
        self.stream.start()

    def read_frames(self, duration_ms: int) -> bytes:
        """
        Read approximately `duration_ms` milliseconds of audio into a single bytes object.
        """
        if self.stream is None:
            raise RuntimeError("VADStream not started")
        frames = []
        total_bytes = int(self.sample_rate * (duration_ms / 1000.0) * 2)
        remaining = total_bytes
        while remaining > 0:
            block = self.stream.read(int(self.frame_bytes / 2))[0].tobytes()
            frames.append(block)
            remaining -= len(block)
        return b"".join(frames)

    def detect_speech_segment(
        self,
        max_silence_ms: int = 600,
        min_speech_ms: int = 200,
        max_utterance_ms: int = 30000,
    ) -> bytes:
        """
        Block until a speech segment is detected followed by sufficient silence,
        then return the raw PCM bytes for that segment.

        Returns after at most `max_utterance_ms`: the loop previously had no
        other exit, so silence from a stuck device blocked the caller forever.
        """
        if self.stream is None:
            raise RuntimeError("VADStream not started")

        speech = bytearray()
        voiced = 0
        unvoiced = 0
        started = False
        deadline = time.perf_counter() + max_utterance_ms / 1000.0
        while True:
            if time.perf_counter() >= deadline:
                break
            data = self.stream.read(int(self.frame_bytes / 2))[0].tobytes()
            if len(data) < self.frame_bytes:
                continue
            is_speech = self.vad.is_speech(data, self.sample_rate)
            if is_speech:
                voiced += 1
                unvoiced = 0
                speech.extend(data)
                if not started and voiced * 30 >= min_speech_ms:
                    started = True
            else:
                if started:
                    unvoiced += 1
                    if unvoiced * 30 >= max_silence_ms:
                        break
                else:
                    voiced = 0
        return bytes(speech)
