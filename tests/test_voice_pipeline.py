import re
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pytest

from src.simple_voice_handler import SimpleVoiceHandler
from src.state_manager import ConversationState
from src.voice_client import VoiceClient


class _FakeASR:
    def __init__(self, text: str = "hello from user") -> None:
        self.text = text

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> tuple[str, float]:
        assert isinstance(wav_bytes, (bytes, bytearray))
        return self.text, 123.0

    def streaming_listen(
        self, vad_stream, on_partial: Callable[[str], None] | None = None, **_: Any
    ) -> tuple[str, float, float]:
        if on_partial:
            on_partial(self.text)
        return self.text, 111.0, 1.0


class _FakeLLM:
    def __init__(self, text: str = "assistant reply.") -> None:
        self.text = text

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, float, dict[str, Any] | None]:
        assert messages and messages[0]["role"] == "system"
        return self.text, 222.0, None

    def stream_chat(self, messages: list[dict[str, str]]):
        assert messages and messages[0]["role"] == "system"

        def gen() -> Iterable[str]:
            # Stream the configured text, not a fixed phrase: the handler now
            # builds the assistant turn from the stream, so a fake that ignores
            # its own reply text would make every streaming assertion vacuous.
            yield from re.findall(r"\S+\s*", self.text) or [self.text]

        return gen(), 200.0


class _FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize_sentence(self, text: str) -> bytes:
        self.spoken.append(text)
        return b"fake-wav"

    def speak_sentences(self, sentences: Iterable[str], stop_flag: Callable[[], bool]) -> float:
        for s in sentences:
            if stop_flag():
                break
            self.spoken.append(s)
        return 333.0

    class playback:  # type: ignore[override]
        @staticmethod
        def play_wav(wav_bytes: bytes) -> None:
            assert isinstance(wav_bytes, (bytes, bytearray))


class _FakeLogger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def log_turn(
        self,
        turn: int,
        persona: str,
        asr_ms: float,
        llm_ms: float,
        tts_ms: float,
        total_ms: float,
        *_,
        **__,
    ):
        self.rows.append(
            {
                "turn": turn,
                "persona": persona,
                "asr_ms": asr_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": total_ms,
            }
        )


def test_simple_voice_handler_happy_path(monkeypatch) -> None:
    persona = {"name": "Test Persona", "system_prompt": "Be helpful."}
    callbacks: dict[str, Any] = {"status": [], "user_text": [], "assistant_text": [], "metrics": []}

    def _append_cb(key: str):
        def inner(value):
            callbacks[key].append(value)

        return inner

    handler = SimpleVoiceHandler(
        persona,
        callbacks={
            "status": _append_cb("status"),
            "user_text": _append_cb("user_text"),
            "assistant_text": _append_cb("assistant_text"),
            "metrics": _append_cb("metrics"),
        },
        asr_client=_FakeASR("hello world"),
        llm_client=_FakeLLM("hi there."),
        tts_client=_FakeTTS(),
    )

    # Seed some fake audio data in the buffer so that process_voice_input
    # does not attempt to read from a real microphone.
    fake_chunk = np.zeros((1600, 1), dtype=np.int16)
    handler.audio_buffer.put(fake_chunk)

    metrics = handler.process_voice_input()

    assert "error" not in metrics
    assert metrics["asr_ms"] == 123.0
    assert metrics["llm_ms"] >= 0
    assert metrics["tts_ms"] >= 0
    # total_ms is measured wall-clock for the turn, while asr_ms here is a
    # fixed 123.0 reported by a stub that returns instantly. Asserting
    # total >= asr compares a real measurement against a fabricated one, and
    # fails for a reason that says nothing about the code. What the test can
    # honestly check is that the total is a real, non-negative measurement and
    # that the stub's value was passed through untouched (asserted above).
    assert metrics["total_ms"] >= 0

    assert callbacks["user_text"][-1] == "hello world"
    assert callbacks["assistant_text"][-1] == "hi there."
    # Streaming is what makes a voice turn feel responsive, so the turn must
    # report when the first token and the first audio actually happened.
    assert metrics["first_token_ms"] is not None
    assert metrics["first_audio_ms"] is not None
    assert callbacks["metrics"], "Expected metrics callback to be invoked"


class _TestVoiceClient(VoiceClient):
    """Subclass VoiceClient to avoid touching real audio devices in tests."""

    def __init__(self, persona: dict[str, Any]) -> None:
        super().__init__(
            persona=persona,
            session_id="test-session",
            callbacks={},
            asr=_FakeASR("hello from user"),
            llm=_FakeLLM("assistant reply."),
            tts=_FakeTTS(),
            vad_stream=None,
        )

    def listen_once(self) -> tuple[str, float, float]:  # type: ignore[override]
        return "hello from user", 111.0, 1.0

    def monitor_barge_in(self) -> None:  # type: ignore[override]
        # No-op in tests
        return

    def stop_barge_in_monitor(self) -> None:  # type: ignore[override]
        return


def test_voice_client_run_turn_logs_and_updates_state(monkeypatch, tmp_path) -> None:
    persona = {"name": "Customer", "system_prompt": "Be concise."}
    vc = _TestVoiceClient(persona=persona)
    logger_obj = _FakeLogger()

    class DummyFeedback:
        @staticmethod
        def evaluate(state: ConversationState) -> str:
            return f"turns={len(state.turns)}"

    vc.run(max_turns=1, logger_obj=logger_obj, feedback=DummyFeedback)

    assert len(logger_obj.rows) == 1
    assert logger_obj.rows[0]["persona"] == "Customer"
    assert len(vc.state.turns) == 2  # user + assistant


def test_the_logged_speaker_is_the_persona_not_a_hardcoded_role() -> None:
    # Every persona casts the model as the support agent, but the CLI printed
    # and logged its replies under the fixed label "Customer" — the other role.
    logger_obj = _FakeLogger()
    vc = _TestVoiceClient(persona={"name": "Lost Card Support", "system_prompt": "p"})
    vc.run(max_turns=1, logger_obj=logger_obj)

    assert logger_obj.rows[0]["persona"] == "Lost Card Support"


# --- SimpleVoiceHandler: the paths that are not the happy one ---------------


def _handler(**overrides) -> SimpleVoiceHandler:
    kwargs: dict[str, Any] = {
        "persona": {"name": "Test Persona", "system_prompt": "Be helpful."},
        "asr_client": _FakeASR("hello world"),
        "llm_client": _FakeLLM("hi there."),
        "tts_client": _FakeTTS(),
    }
    kwargs.update(overrides)
    persona = kwargs.pop("persona")
    return SimpleVoiceHandler(persona, **kwargs)


def test_no_recorded_audio_is_reported_rather_than_transcribed() -> None:
    assert _handler().process_voice_input() == {"error": "No audio recorded"}


def test_silence_is_reported_as_no_speech() -> None:
    handler = _handler(asr_client=_FakeASR("   "))
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))

    assert handler.process_voice_input() == {"error": "No speech detected"}
    # The empty transcript must not enter the conversation as a user turn.
    assert handler.state.turns == []


def test_an_empty_model_reply_is_reported() -> None:
    handler = _handler(llm_client=_FakeLLM("   "))
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))

    assert handler.process_voice_input() == {"error": "No response generated"}


def test_a_failure_mid_pipeline_is_returned_and_announced() -> None:
    class _ExplodingLLM(_FakeLLM):
        def stream_chat(self, messages):
            raise RuntimeError("provider is down")

    errors: list[str] = []
    handler = _handler(llm_client=_ExplodingLLM(), callbacks={"error": errors.append})
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))

    result = handler.process_voice_input()
    assert result["error"] == "provider is down"
    assert errors == ["provider is down"]


def test_the_persona_prompt_reaches_the_model() -> None:
    seen: list[list[dict[str, str]]] = []

    class _Recording(_FakeLLM):
        def stream_chat(self, messages):
            seen.append(messages)
            return super().stream_chat(messages)

    handler = _handler(
        persona={"name": "P", "system_prompt": "SYSTEM PROMPT HERE"},
        llm_client=_Recording("ok."),
    )
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))
    handler.process_voice_input()

    assert seen[0][0] == {"role": "system", "content": "SYSTEM PROMPT HERE"}


def test_each_sentence_of_the_reply_is_synthesised() -> None:
    tts = _FakeTTS()
    handler = _handler(llm_client=_FakeLLM("First part. Second part!"), tts_client=tts)
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))
    handler.process_voice_input()

    assert tts.spoken == ["First part.", "Second part!"]


def test_the_conversation_accumulates_across_turns() -> None:
    handler = _handler()
    for _ in range(2):
        handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))
        handler.process_voice_input()

    assert [t["role"] for t in handler.state.turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_reset_clears_the_conversation() -> None:
    handler = _handler()
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))
    handler.process_voice_input()
    handler.reset_conversation()

    assert handler.state.turns == []


# --- SimpleVoiceHandler: device handling -----------------------------------


class _FakeInputStream:
    def __init__(self, fail_on: str | None = None) -> None:
        self.started = False
        self.stopped = False
        self.closed = False
        self._fail_on = fail_on

    def start(self) -> None:
        if self._fail_on == "start":
            raise OSError("no such device")
        self.started = True

    def stop(self) -> None:
        if self._fail_on == "stop":
            raise OSError("device disappeared")
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _patch_devices(monkeypatch, stream: _FakeInputStream) -> None:
    """Replace the lazy sounddevice proxy so no microphone is opened."""
    import src.simple_voice_handler as mod

    class _FakeSoundDevice:
        @staticmethod
        def InputStream(**kwargs) -> _FakeInputStream:
            return stream

    monkeypatch.setattr(mod, "sd", _FakeSoundDevice())


def test_recording_opens_and_releases_the_input_stream(monkeypatch) -> None:
    stream = _FakeInputStream()
    _patch_devices(monkeypatch, stream)

    handler = _handler()
    handler.start_recording()
    assert handler.is_recording is True
    assert stream.started is True

    handler.stop_recording()
    assert handler.is_recording is False
    assert (stream.stopped, stream.closed) == (True, True)
    assert handler.stream is None


def test_a_device_that_cannot_open_does_not_leave_a_recording_flag_set(
    monkeypatch,
) -> None:
    # `is_recording` was set before the device was opened and never cleared on
    # failure, so the UI showed a live recording against a stream that had
    # never started and could never produce audio.
    stream = _FakeInputStream(fail_on="start")
    _patch_devices(monkeypatch, stream)

    errors: list[str] = []
    handler = _handler(callbacks={"error": errors.append})
    handler.start_recording()

    assert handler.is_recording is False
    assert handler.stream is None
    assert errors


def test_a_stream_that_fails_to_stop_is_still_released(monkeypatch) -> None:
    # A raising stop() used to leave self.stream set and the device open, so
    # the next start_recording() opened a second stream on top of it.
    stream = _FakeInputStream(fail_on="stop")
    _patch_devices(monkeypatch, stream)

    handler = _handler()
    handler.start_recording()
    handler.stop_recording()

    assert handler.stream is None
    assert stream.closed is True


def test_cleanup_releases_playback_even_when_the_device_fails(monkeypatch) -> None:
    stream = _FakeInputStream(fail_on="stop")
    _patch_devices(monkeypatch, stream)

    stopped: list[bool] = []

    class _TTS(_FakeTTS):
        class playback:  # type: ignore[override]
            @staticmethod
            def stop() -> None:
                stopped.append(True)

    handler = _handler(tts_client=_TTS())
    handler.start_recording()
    handler.cleanup()

    assert stopped == [True]
    assert handler.stream is None


def test_the_audio_callback_only_buffers_while_recording() -> None:
    handler = _handler()
    chunk = np.zeros((160, 1), dtype=np.int16)

    handler._audio_callback(chunk, 160, None, None)
    assert handler.audio_buffer.empty()

    handler.is_recording = True
    handler._audio_callback(chunk, 160, None, None)
    assert not handler.audio_buffer.empty()


def test_starting_a_recording_discards_the_previous_buffer(monkeypatch) -> None:
    _patch_devices(monkeypatch, _FakeInputStream())

    handler = _handler()
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))
    handler.start_recording()

    assert handler.audio_buffer.empty()


# --- VoiceClient -----------------------------------------------------------


def test_starting_without_a_vad_stream_is_a_no_op() -> None:
    # Text mode and the tests both construct the client with audio disabled;
    # start() must not try to open a device or raise.
    vc = _TestVoiceClient(persona={"name": "C", "system_prompt": "p"})
    vc.start()
    assert vc.vad_stream is None


def test_listening_without_audio_explains_itself() -> None:
    # The None used to be handed to streaming_listen and surfaced as an
    # AttributeError on `.stream`, which says nothing about the cause.
    vc = VoiceClient(
        persona={"name": "C", "system_prompt": "p"},
        session_id="s",
        asr=_FakeASR(),
        llm=_FakeLLM(),
        tts=_FakeTTS(),
        vad_stream=None,
    )
    with pytest.raises(RuntimeError, match="cannot listen"):
        vc.listen_once()


def test_an_explicit_none_vad_stream_is_respected() -> None:
    # `x or Default()` could not tell "omitted" from "explicitly None", so a
    # test opting out of audio still got a real VADStream — and a microphone.
    vc = VoiceClient(
        persona={"name": "C", "system_prompt": "p"},
        session_id="s",
        asr=_FakeASR(),
        llm=_FakeLLM(),
        tts=_FakeTTS(),
        vad_stream=None,
    )
    assert vc.vad_stream is None


def test_a_supplied_vad_stream_is_used_as_given() -> None:
    sentinel = object()
    vc = VoiceClient(
        persona={"name": "C", "system_prompt": "p"},
        session_id="s",
        asr=_FakeASR(),
        llm=_FakeLLM(),
        tts=_FakeTTS(),
        vad_stream=sentinel,  # type: ignore[arg-type]
    )
    assert vc.vad_stream is sentinel


def test_the_assistant_reply_is_spoken_and_recorded() -> None:
    tts = _FakeTTS()
    vc = _TestVoiceClient(persona={"name": "C", "system_prompt": "Be concise."})
    vc.tts = tts
    vc.run(max_turns=1, logger_obj=_FakeLogger())

    assert tts.spoken == ["assistant reply."]
    assert vc.state.turns[-1] == {"role": "assistant", "text": "assistant reply."}


def test_metrics_are_emitted_for_every_turn() -> None:
    emitted: list[dict[str, Any]] = []
    vc = _TestVoiceClient(persona={"name": "C", "system_prompt": "p"})
    vc.callbacks = {"turn_metrics": emitted.append}
    vc.run(max_turns=2, logger_obj=_FakeLogger())

    assert [m["turn"] for m in emitted] == [1, 2]
    for metrics in emitted:
        assert metrics["total_ms"] == pytest.approx(
            metrics["asr_ms"] + metrics["llm_ms"] + metrics["tts_ms"]
        )


def test_a_failing_callback_does_not_break_the_turn() -> None:
    vc = _TestVoiceClient(persona={"name": "C", "system_prompt": "p"})
    vc.callbacks = {"status": lambda *_a, **_k: (_ for _ in ()).throw(ValueError("ui gone"))}
    vc.run(max_turns=1, logger_obj=_FakeLogger())

    assert len(vc.state.turns) == 2


def test_a_requested_stop_ends_the_run_before_the_next_turn() -> None:
    logger_obj = _FakeLogger()
    vc = _TestVoiceClient(persona={"name": "C", "system_prompt": "p"})
    vc.request_stop()
    vc.run(max_turns=3, logger_obj=logger_obj)

    assert logger_obj.rows == []


def test_the_cost_estimate_follows_the_configured_prices(monkeypatch) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setenv("LLM_PRICE_IN_PER_1K", "1")
    monkeypatch.setenv("LLM_PRICE_OUT_PER_1K", "2")

    vc = _TestVoiceClient(persona={"name": "C", "system_prompt": "p"})
    vc.callbacks = {"turn_metrics": emitted.append}
    vc.run(max_turns=1, logger_obj=_FakeLogger())

    metrics = emitted[0]
    expected = metrics["tokens_in"] / 1000.0 + 2 * metrics["tokens_out"] / 1000.0
    assert metrics["cost_est"] == pytest.approx(expected)


def test_an_unparseable_price_does_not_break_the_turn(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PRICE_IN_PER_1K", "free")
    vc = _TestVoiceClient(persona={"name": "C", "system_prompt": "p"})
    vc.run(max_turns=1, logger_obj=_FakeLogger())

    assert len(vc.state.turns) == 2


def test_feedback_runs_over_the_finished_conversation() -> None:
    seen: list[int] = []

    class _Feedback:
        @staticmethod
        def evaluate(state: ConversationState) -> str:
            seen.append(len(state.turns))
            return "done"

    vc = _TestVoiceClient(persona={"name": "C", "system_prompt": "p"})
    vc.run(max_turns=2, logger_obj=_FakeLogger(), feedback=_Feedback)

    assert seen == [4]


# --- SimpleVoiceHandler: streaming and barge-in -----------------------------


def test_synthesis_starts_before_the_model_has_finished_writing() -> None:
    # This is the whole latency argument: the first sentence must reach the
    # speaker while the model is still producing the second.
    order: list[str] = []

    class _SlowLLM(_FakeLLM):
        def stream_chat(self, messages):
            def gen():
                yield "First sentence. "
                order.append("second-sentence-produced")
                yield "Second sentence."

            return gen(), 0.0

    class _RecordingTTS(_FakeTTS):
        def speak_sentences(self, sentences, stop_flag):
            for sentence in sentences:
                order.append(f"spoke:{sentence}")
            return 1.0

    handler = _handler(llm_client=_SlowLLM(), tts_client=_RecordingTTS())
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))
    handler.process_voice_input()

    assert order == [
        "spoke:First sentence.",
        "second-sentence-produced",
        "spoke:Second sentence.",
    ]


def test_an_interrupted_turn_stops_speaking() -> None:
    # Barge-in arrives mid-turn, from whatever is watching the microphone, and
    # must abandon the rest of the reply rather than finish it.
    class _InterruptingTTS(_FakeTTS):
        def __init__(self, handler_box) -> None:
            super().__init__()
            self._box = handler_box

        def speak_sentences(self, sentences, stop_flag):
            for sentence in sentences:
                if stop_flag():
                    break
                self.spoken.append(sentence)
                self._box[0].interrupt()
            return 1.0

    box: list[Any] = [None]
    tts = _InterruptingTTS(box)
    handler = _handler(llm_client=_FakeLLM("One. Two. Three."), tts_client=tts)
    box[0] = handler
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))
    handler.process_voice_input()

    assert tts.spoken == ["One."]


def test_a_new_recording_counts_as_an_interruption() -> None:
    handler = _handler()
    assert handler._should_stop_speaking() is False

    handler.is_recording = True
    assert handler._should_stop_speaking() is True


def test_the_interrupt_flag_is_cleared_at_the_start_of_a_turn() -> None:
    # Otherwise one barge-in would silence every turn that followed it.
    handler = _handler()
    handler.interrupt()
    handler.audio_buffer.put(np.zeros((160, 1), dtype=np.int16))
    metrics = handler.process_voice_input()

    assert "error" not in metrics
    assert handler._interrupt.is_set() is False
