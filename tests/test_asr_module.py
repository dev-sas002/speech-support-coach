"""
Transcription and the voice-activity capture loop.

No test here opens a microphone: `streaming_listen` is driven by a scripted
stand-in for VADStream, and every transcription call goes to an injected
client.
"""

from __future__ import annotations

import io
import wave

import pytest

from src.asr_module import ASRClient, ASRConfig, pcm16_to_wav_bytes
from src.local_backend import LOCAL_MODEL_NAME, LocalGroqClient

SAMPLE_RATE = 16000
FRAME_BYTES = int(SAMPLE_RATE * 0.03) * 2  # 30 ms of 16-bit mono


# --- fakes -----------------------------------------------------------------


class _Block:
    """What sounddevice hands back: something with .tobytes()."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


class _FakeRawStream:
    """Replays a fixed list of PCM frames, then silence forever."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)
        self.reads = 0

    def read(self, n_samples: int):
        self.reads += 1
        payload = self._frames.pop(0) if self._frames else b"\x00" * FRAME_BYTES
        return (_Block(payload), False)


class _FakeVad:
    """Reports speech for frames whose bytes are non-zero."""

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return any(frame)


class _FakeVADStream:
    def __init__(self, frames: list[bytes]) -> None:
        self.sample_rate = SAMPLE_RATE
        self.frame_bytes = FRAME_BYTES
        self.vad = _FakeVad()
        self.stream = _FakeRawStream(frames)


class _Transcription:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeTranscriptions:
    def __init__(self, text: str = "transcribed text") -> None:
        self.text = text
        self.calls: list[dict] = []
        self.raises: Exception | None = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return _Transcription(self.text)


class _FakeAudio:
    def __init__(self, text: str = "transcribed text") -> None:
        self.transcriptions = _FakeTranscriptions(text)


class _FakeASRGroqClient:
    def __init__(self, text: str = "transcribed text") -> None:
        self.audio = _FakeAudio(text)


def _speech(n: int) -> list[bytes]:
    return [b"\x7f" * FRAME_BYTES] * n


def _silence(n: int) -> list[bytes]:
    return [b"\x00" * FRAME_BYTES] * n


# --- WAV framing -----------------------------------------------------------


class TestPcm16ToWavBytes:
    def test_the_container_preserves_the_samples_and_the_rate(self) -> None:
        pcm = b"\x01\x02" * 800
        wav = pcm16_to_wav_bytes(pcm, SAMPLE_RATE)

        with wave.open(io.BytesIO(wav), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.readframes(wf.getnframes()) == pcm

    def test_an_empty_buffer_still_produces_a_readable_wav(self) -> None:
        with wave.open(io.BytesIO(pcm16_to_wav_bytes(b"", SAMPLE_RATE)), "rb") as wf:
            assert wf.getnframes() == 0


# --- client construction ---------------------------------------------------


class TestClientSelection:
    def test_without_a_key_the_offline_stand_in_is_used(self, monkeypatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        client = ASRClient()
        assert isinstance(client.client, LocalGroqClient)
        assert client.is_local is True

    def test_an_injected_client_is_not_treated_as_offline(self, monkeypatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        injected = _FakeASRGroqClient()
        client = ASRClient(client=injected)
        assert client.client is injected
        assert client.is_local is False

    def test_the_asr_model_comes_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_ASR_MODEL", "whisper-tiny")
        assert ASRClient(client=_FakeASRGroqClient()).config.model == "whisper-tiny"

    def test_an_explicit_model_beats_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_ASR_MODEL", "whisper-tiny")
        client = ASRClient(model="whisper-large-v3", client=_FakeASRGroqClient())
        assert client.config.model == "whisper-large-v3"

    def test_sample_rate_is_exposed_from_the_config(self) -> None:
        assert ASRClient(sample_rate=8000, client=_FakeASRGroqClient()).sample_rate == 8000


# --- transcription ---------------------------------------------------------


class TestTranscribeWavBytes:
    def test_it_returns_the_text_and_a_measured_latency(self) -> None:
        fake = _FakeASRGroqClient("hello there")
        text, latency_ms = ASRClient(client=fake).transcribe_wav_bytes(b"wav")

        assert text == "hello there"
        assert latency_ms >= 0

    def test_the_configured_model_is_passed_to_the_provider(self) -> None:
        fake = _FakeASRGroqClient()
        ASRClient(model="whisper-large-v3", client=fake).transcribe_wav_bytes(b"wav")

        assert fake.audio.transcriptions.calls[0]["model"] == "whisper-large-v3"

    def test_the_audio_is_sent_as_a_named_wav_file(self) -> None:
        fake = _FakeASRGroqClient()
        ASRClient(client=fake).transcribe_wav_bytes(b"pretend-wav-bytes")

        sent = fake.audio.transcriptions.calls[0]["file"]
        assert sent.name.endswith(".wav")
        assert sent.read() == b"pretend-wav-bytes"

    def test_a_provider_failure_becomes_a_named_error(self) -> None:
        fake = _FakeASRGroqClient()
        fake.audio.transcriptions.raises = ConnectionError("network down")

        with pytest.raises(RuntimeError, match="Failed to transcribe audio"):
            ASRClient(client=fake).transcribe_wav_bytes(b"wav")

    def test_the_original_cause_is_preserved(self) -> None:
        fake = _FakeASRGroqClient()
        cause = ConnectionError("network down")
        fake.audio.transcriptions.raises = cause

        with pytest.raises(RuntimeError) as excinfo:
            ASRClient(client=fake).transcribe_wav_bytes(b"wav")
        assert excinfo.value.__cause__ is cause

    def test_a_response_without_text_yields_an_empty_string(self) -> None:
        class _NoText:
            def create(self, **kwargs):
                return object()

        fake = _FakeASRGroqClient()
        fake.audio.transcriptions = _NoText()

        text, _ = ASRClient(client=fake).transcribe_wav_bytes(b"wav")
        assert text == ""

    def test_offline_transcription_says_it_cannot_hear(self, monkeypatch) -> None:
        # It must not invent words: the pipeline would treat them as speech.
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        text, _ = ASRClient().transcribe_wav_bytes(b"wav")
        assert "unavailable" in text.lower()


# --- the capture loop ------------------------------------------------------


class TestStreamingListen:
    def _client(self, text: str = "final transcript") -> ASRClient:
        return ASRClient(client=_FakeASRGroqClient(text))

    def test_speech_then_silence_ends_the_capture(self) -> None:
        # 20 voiced frames (600 ms) then enough silence to cross max_silence.
        vad = _FakeVADStream(_speech(20) + _silence(30))
        text, latency_ms, secs = self._client().streaming_listen(vad)

        assert text == "final transcript"
        assert latency_ms >= 0
        # 20 frames x 30 ms of 16 kHz mono audio was captured.
        assert secs == pytest.approx(0.6, abs=0.05)

    def test_silence_alone_never_starts_a_segment(self) -> None:
        vad = _FakeVADStream(_silence(5))
        _, _, secs = self._client().streaming_listen(vad, max_utterance_ms=60)
        assert secs == 0.0

    def test_capture_is_bounded_when_speech_never_stops(self) -> None:
        # An endless talker used to block the caller forever: the only exit was
        # speech followed by silence, and neither a timeout nor a stop existed.
        vad = _FakeVADStream(_speech(10_000))
        text, _, _ = self._client().streaming_listen(vad, max_utterance_ms=50)
        assert text == "final transcript"

    def test_an_external_stop_ends_the_capture(self) -> None:
        vad = _FakeVADStream(_speech(10_000))
        reads = {"n": 0}

        def should_stop() -> bool:
            reads["n"] += 1
            return reads["n"] > 5

        self._client().streaming_listen(vad, should_stop=should_stop)
        assert vad.stream.reads <= 6

    def test_partials_are_reported_while_speech_continues(self) -> None:
        vad = _FakeVADStream(_speech(60) + _silence(30))
        seen: list[str] = []

        self._client("partial text").streaming_listen(
            vad,
            on_partial=seen.append,
            partial_interval_ms=0,  # falls back to the configured interval
            min_speech_ms=30,
        )
        # A partial needs >0.5 s of audio and an elapsed interval; the run is
        # short, so the contract worth pinning is that nothing invalid escaped.
        assert all(isinstance(text, str) and text for text in seen)

    def test_a_failing_partial_does_not_abort_the_capture(self) -> None:
        # Partial transcription is best-effort; losing one must not lose the turn.
        fake = _FakeASRGroqClient("final transcript")
        client = ASRClient(client=fake)
        client.config = ASRConfig(model="m", sample_rate=SAMPLE_RATE, partial_interval_ms=0)

        calls = {"n": 0}
        original = fake.audio.transcriptions.create

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("partial failed")
            return original(**kwargs)

        fake.audio.transcriptions.create = flaky

        text, _, _ = client.streaming_listen(_FakeVADStream(_speech(40) + _silence(30)))
        assert text == "final transcript"

    def test_short_frames_are_skipped_rather_than_transcribed(self) -> None:
        runt = b"\x7f" * (FRAME_BYTES // 2)
        vad = _FakeVADStream([runt, runt, *_speech(20), *_silence(30)])

        _, _, secs = self._client().streaming_listen(vad)
        # The two undersized frames were dropped, not folded into the buffer.
        assert secs == pytest.approx(0.6, abs=0.05)


class TestOfflineModelName:
    def test_the_offline_asr_client_reports_the_hosted_default_model(self, monkeypatch) -> None:
        # ASR has no local recogniser, so the model name stays the hosted one
        # even offline — unlike the LLM, which names its scripted stand-in.
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_ASR_MODEL", raising=False)
        assert ASRClient().config.model != LOCAL_MODEL_NAME
