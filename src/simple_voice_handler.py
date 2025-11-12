"""
Push-to-talk capture for the Streamlit UI.

The handler owns the microphone and the turn it produces: record, transcribe,
stream a reply, speak it. It deliberately does *not* own the conversation
policy — the persona, the window, the prompt — which lives in
`ConversationState` and is shared with every other entry point.

The reply is streamed and spoken sentence by sentence rather than waited for in
full. That is the difference between a turn that starts answering in a few
hundred milliseconds and one that starts answering when the model has finished
writing, and on a slow provider it is the difference between a usable agent and
an unusable one.
"""

import contextlib
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger

from .asr_module import ASRClient, pcm16_to_wav_bytes
from .conversation.streaming import split_sentences
from .llm_module import LLMClient
from .state_manager import ConversationState
from .tts_module import KokoroTTSClient


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


sd = _LazySoundDevice()


class SimpleVoiceHandler:
    """
    Simplified voice handler with manual recording controls.

    Dependencies (ASR, LLM, TTS) can be injected for testing.
    """

    def __init__(
        self,
        persona: dict[str, Any],
        callbacks: dict[str, Callable] | None = None,
        asr_client: ASRClient | None = None,
        llm_client: LLMClient | None = None,
        tts_client: KokoroTTSClient | None = None,
    ) -> None:
        self.persona = persona
        self.callbacks: dict[str, Callable] = callbacks or {}

        self.asr_client = asr_client or ASRClient()
        self.llm_client = llm_client or LLMClient()
        self.tts_client = tts_client or KokoroTTSClient()

        self.state = ConversationState(
            session_id="streamlit_session",
            persona_name=persona.get("name", "Assistant"),
        )

        self.sample_rate = 16000
        self.audio_buffer: queue.Queue = queue.Queue()
        self.stream: Any | None = None  # sounddevice.InputStream
        self.is_recording = False
        #: Set by `interrupt()` to abandon the reply in progress. A front end
        #: that can run concurrently with a turn (the CLI client does; a
        #: Streamlit rerun does not) uses this for barge-in.
        self._interrupt = threading.Event()

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # type: ignore[override]
        if self.is_recording:
            self.audio_buffer.put(indata.copy())

    def start_recording(self) -> None:
        """Start recording audio."""
        try:
            while not self.audio_buffer.empty():
                self.audio_buffer.get()

            self.is_recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
            self.stream.start()

            if self.callbacks.get("status"):
                self.callbacks["status"]("🎤 Recording...")

            logger.info("Recording started")
        except Exception as e:
            # `is_recording` was set before the device was opened and never
            # cleared on failure, so the UI showed a live recording against a
            # stream that had never started and could never produce audio.
            self.is_recording = False
            self._close_stream()
            logger.error(f"Error starting recording: {e}")
            if self.callbacks.get("error"):
                self.callbacks["error"](str(e))

    def _close_stream(self) -> None:
        """Release the input stream, whatever state the device is in."""
        stream, self.stream = self.stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as exc:  # the device may already be gone
            logger.warning(f"Error stopping input stream: {exc}")
        try:
            stream.close()
        except Exception as exc:
            logger.warning(f"Error closing input stream: {exc}")

    def stop_recording(self) -> None:
        """Stop recording audio."""
        self.is_recording = False
        # A raising stop() used to leave self.stream set and the device open,
        # so the next start_recording() opened a second stream on top of it.
        self._close_stream()
        logger.info("Recording stopped")

    def _emit(self, key: str, *args, **kwargs) -> None:
        cb = self.callbacks.get(key)
        if cb:
            # A front end that has gone away must not take the turn with it.
            with contextlib.suppress(Exception):
                cb(*args, **kwargs)

    def interrupt(self) -> None:
        """Abandon the reply currently being spoken."""
        self._interrupt.set()

    def _should_stop_speaking(self) -> bool:
        # Starting a new recording is itself an interruption: the caller has
        # begun talking over the agent.
        return self._interrupt.is_set() or self.is_recording

    def process_voice_input(self) -> dict[str, Any]:
        """
        Run one turn: transcribe the recording, stream a reply, speak it.

        Returns a metrics dict, or a dict with an ``error`` key on failure.

        The stages overlap by design, so the timings are not additive:
        `llm_ms` and `tts_ms` both run from the moment the request opened and
        cover the same wall clock. `first_audio_ms` is the honest measure of
        responsiveness — when the caller first heard something.
        """
        metrics: dict[str, Any] = {}
        start_total = time.perf_counter()
        self._interrupt.clear()

        try:
            self.stop_recording()
            self._emit("status", "🎯 Transcribing...")

            audio_chunks = []
            while not self.audio_buffer.empty():
                audio_chunks.append(self.audio_buffer.get())

            if not audio_chunks:
                return {"error": "No audio recorded"}

            audio_data = np.concatenate(audio_chunks, axis=0)
            wav_bytes = pcm16_to_wav_bytes(audio_data.flatten().tobytes(), self.sample_rate)
            user_text, asr_ms = self.asr_client.transcribe_wav_bytes(wav_bytes)
            metrics["asr_ms"] = asr_ms

            if not user_text or not user_text.strip():
                return {"error": "No speech detected"}

            logger.info(f"User said: {user_text}")
            self.state.add_turn("user", user_text)
            self._emit("user_text", user_text)

            self._emit("status", "🤖 Thinking...")
            self._emit("llm_start", user_text)

            messages = self.state.as_messages(
                self.persona.get("system_prompt", "You are a helpful assistant."),
            )

            llm_start = time.perf_counter()
            token_stream, _ = self.llm_client.stream_chat(messages)

            first_token_ms: list[float | None] = [None]
            first_audio_ms: list[float | None] = [None]
            llm_done: list[float | None] = [None]
            pieces: list[str] = []

            def timed_tokens():
                for token in token_stream:
                    if first_token_ms[0] is None:
                        first_token_ms[0] = (time.perf_counter() - llm_start) * 1000
                    yield token
                llm_done[0] = time.perf_counter()

            def on_partial(token: str) -> None:
                pieces.append(token)
                self._emit("assistant_partial", token)

            sentences = split_sentences(
                timed_tokens(),
                stop_flag=self._should_stop_speaking,
                on_partial=on_partial,
            )

            def timed_sentences():
                for sentence in sentences:
                    if first_audio_ms[0] is None:
                        first_audio_ms[0] = (time.perf_counter() - llm_start) * 1000
                    self._emit("assistant_sentence", sentence)
                    yield sentence

            self._emit("status", "🔊 Speaking...")
            tts_ms = self.tts_client.speak_sentences(timed_sentences(), self._should_stop_speaking)

            assistant_text = "".join(pieces).strip()
            metrics["llm_ms"] = ((llm_done[0] or time.perf_counter()) - llm_start) * 1000
            metrics["tts_ms"] = tts_ms
            metrics["first_token_ms"] = first_token_ms[0]
            metrics["first_audio_ms"] = first_audio_ms[0]

            if not assistant_text:
                return {"error": "No response generated"}

            logger.info(f"Assistant said: {assistant_text}")
            self.state.add_turn("assistant", assistant_text)
            self._emit("assistant_text", assistant_text)

            metrics["total_ms"] = (time.perf_counter() - start_total) * 1000
            logger.info(
                "Metrics: ASR={asr:.0f}ms, first token={ttft}, first audio={tta}, "
                "total={total:.0f}ms".format(
                    asr=asr_ms,
                    ttft=("n/a" if first_token_ms[0] is None else f"{first_token_ms[0]:.0f}ms"),
                    tta=("n/a" if first_audio_ms[0] is None else f"{first_audio_ms[0]:.0f}ms"),
                    total=metrics["total_ms"],
                )
            )

            self._emit("metrics", metrics)
            return metrics

        except Exception as exc:
            logger.error(f"Error processing voice input: {exc}")
            self._emit("error", str(exc))
            return {"error": str(exc)}

    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.state.turns = []
        logger.info("Conversation reset")

    def cleanup(self) -> None:
        """Clean up resources."""
        # Playback was released only if closing the input stream succeeded.
        self.is_recording = False
        try:
            self._close_stream()
        finally:
            self.tts_client.playback.stop()
        logger.info("Cleanup complete")
