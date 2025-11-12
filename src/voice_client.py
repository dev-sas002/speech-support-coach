import contextlib
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from loguru import logger

from .asr_module import ASRClient, VADStream
from .conversation.streaming import split_sentences
from .llm_module import LLMClient, approx_tokens
from .state_manager import ConversationState
from .tts_module import KokoroTTSClient

#: Re-exported: sentence segmentation moved to the conversation package when
#: the push-to-talk handler started needing it too. Imports of
#: `src.voice_client.split_sentences` keep working.
__all__ = ["AudioCapture", "VoiceClient", "split_sentences"]


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


#: Distinguishes "argument omitted" from "argument explicitly None".
_NOT_GIVEN: Any = object()


class AudioCapture:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30) -> None:
        self.sample_rate = sample_rate
        self.blocksize = int(sample_rate * (frame_ms / 1000.0))
        self.q: queue.Queue = queue.Queue()
        self.stream: Any | None = None  # sounddevice.InputStream

    def _cb(self, indata, frames, time_info, status) -> None:  # type: ignore[override]
        self.q.put(indata.copy())

    def start(self) -> None:
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.blocksize,
            callback=self._cb,
        )
        self.stream.start()

    def read(self):
        return self.q.get()


class VoiceClient:
    """
    Full duplex voice client for CLI use, orchestrating ASR → LLM → TTS with barge-in.
    """

    def __init__(
        self,
        persona: dict,
        session_id: str,
        callbacks: dict[str, Callable] | None = None,
        asr: ASRClient | None = None,
        llm: LLMClient | None = None,
        tts: KokoroTTSClient | None = None,
        vad_stream: VADStream | None = _NOT_GIVEN,  # type: ignore[assignment]
    ) -> None:
        self.sample_rate = 16000
        self.persona = persona
        self.state = ConversationState(
            session_id=session_id, persona_name=persona.get("name", "Assistant")
        )
        self.asr = asr or ASRClient(sample_rate=self.sample_rate)
        self.llm = llm or LLMClient()
        self.tts = tts or KokoroTTSClient()
        # `x or Default()` cannot tell "not provided" from "explicitly None",
        # so passing vad_stream=None to opt out of audio still constructed a
        # real VADStream — which binds the microphone. A sentinel lets an
        # explicit None mean what it says.
        self.vad_stream = (
            VADStream(sample_rate=self.sample_rate) if vad_stream is _NOT_GIVEN else vad_stream
        )
        self.barge_in_flag = threading.Event()
        self.stop_event = threading.Event()
        self.callbacks: dict[str, Callable] = callbacks or {}

    def emit(self, name: str, *args, **kwargs) -> None:
        cb = self.callbacks.get(name)
        if cb:
            # A failing UI callback must not abort the turn it is reporting on.
            with contextlib.suppress(Exception):
                cb(*args, **kwargs)

    def start(self) -> None:
        # No VAD stream means audio capture was deliberately disabled (text
        # mode, or a test). Starting is then a no-op rather than an error.
        if self.vad_stream is None:
            return
        self.vad_stream.start()

    def listen_once(self) -> tuple[str, float, float]:
        if self.vad_stream is None:
            # Without this the None was handed to streaming_listen and surfaced
            # as an AttributeError on `.stream`, which says nothing about the
            # actual cause: this client was built with audio capture disabled.
            raise RuntimeError(
                "This VoiceClient was created without a VAD stream, so it "
                "cannot listen. Use text mode, or construct it with audio "
                "capture enabled."
            )

        partial_last = [0.0]

        def on_partial(text: str) -> None:
            now = time.perf_counter()
            if (now - partial_last[0]) * 1000 >= 400:
                print(f"ASR partial: {text}")
                partial_last[0] = now
                self.emit("asr_partial", text)

        final_text, asr_ms, asr_secs = self.asr.streaming_listen(
            self.vad_stream,
            on_partial=on_partial,
            # request_stop() previously set an event nothing in the listen loop
            # ever read, so a stop could not interrupt the turn in progress.
            should_stop=self.stop_event.is_set,
        )
        self.emit("asr_final", final_text)
        return final_text, asr_ms, asr_secs

    def monitor_barge_in(self) -> None:
        self.barge_in_flag.clear()

        # Barge-in detection is listening for the caller to interrupt, which
        # requires a microphone. Without one there is nothing to monitor.
        if self.vad_stream is None:
            return

        def run() -> None:
            vad = self.vad_stream.vad
            frame_bytes = self.vad_stream.frame_bytes
            streak = 0
            threshold_frames = 5  # ~150ms at 30ms per frame
            while not self.barge_in_flag.is_set():
                data = self.vad_stream.stream.read(int(frame_bytes / 2))[0].tobytes()  # type: ignore[union-attr]
                if len(data) < frame_bytes:
                    continue
                if vad.is_speech(data, self.sample_rate):
                    streak += 1
                else:
                    streak = 0
                if streak >= threshold_frames:
                    self.barge_in_flag.set()
                    break

        th = threading.Thread(target=run, daemon=True)
        th.start()

    def stop_barge_in_monitor(self) -> None:
        self.barge_in_flag.set()

    def run_turn(self, system_prompt: str, turn_idx: int, logger_obj, live_hints=None) -> None:
        logger.info("Listening...")
        self.emit("status", "Listening")
        user_text, asr_ms, asr_secs = self.listen_once()
        print(f"You: {user_text}")
        self.state.add_turn("user", user_text)

        # Every persona casts the model as the support agent and the person at
        # the microphone as the customer, but these lines printed the model's
        # replies under "Customer" — the opposite role, and the opposite of
        # what the UI shows. The persona's own name is the honest label.
        speaker = self.persona.get("name", "Assistant")

        msgs = self.state.as_messages(system_prompt)
        stream, _ = self.llm.stream_chat(msgs)
        t0_llm = time.perf_counter()
        llm_done_time = [None]

        def token_stream_with_done():
            yield from stream
            llm_done_time[0] = time.perf_counter()

        self.monitor_barge_in()
        print(f"{speaker} (streaming): ", end="", flush=True)
        output_sents: list[str] = []

        def on_llm_partial(tok: str) -> None:
            print(tok, end="", flush=True)
            self.emit("llm_partial", tok)

        sentences = split_sentences(
            token_stream_with_done(),
            stop_flag=lambda: self.barge_in_flag.is_set(),
            on_partial=on_llm_partial,
        )

        def sentences_with_capture():
            for s in sentences:
                output_sents.append(s)
                self.emit("assistant_sentence", s)
                yield s

        def stop_flag() -> bool:
            return self.barge_in_flag.is_set()

        self.emit("status", "Speaking")
        tts_ms = self.tts.speak_sentences(sentences_with_capture(), stop_flag)
        print("")
        self.stop_barge_in_monitor()
        end_ref = llm_done_time[0] or time.perf_counter()
        llm_ms = (end_ref - t0_llm) * 1000

        output_text = " ".join(output_sents).strip()
        self.state.add_turn("assistant", output_text)
        if output_text:
            print(f"{speaker} (final): {output_text}")
            self.emit("assistant_final", output_text)

        cost_est = None
        tokens_in = approx_tokens(user_text + "\n" + system_prompt)
        tokens_out = approx_tokens(output_text)
        try:
            price_in = float(os.getenv("LLM_PRICE_IN_PER_1K", "0") or 0)
            price_out = float(os.getenv("LLM_PRICE_OUT_PER_1K", "0") or 0)
            cost_est = (tokens_in / 1000.0) * price_in + (tokens_out / 1000.0) * price_out
        except Exception:
            cost_est = None
        total_ms = asr_ms + llm_ms + tts_ms
        logger_obj.log_turn(
            turn_idx,
            speaker,
            asr_ms,
            llm_ms,
            tts_ms,
            total_ms,
            len(user_text),
            len(output_text),
            tokens_in,
            tokens_out,
            asr_secs,
            len(output_text),
            cost_est,
            None,
        )
        self.emit(
            "turn_metrics",
            {
                "turn": turn_idx,
                "asr_ms": asr_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": total_ms,
                "input_chars": len(user_text),
                "output_chars": len(output_text),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "asr_secs": asr_secs,
                "tts_chars": len(output_text),
                "cost_est": cost_est,
            },
        )

    def run(self, max_turns: int, logger_obj, feedback=None) -> None:
        self.start()
        system_prompt = self.persona.get("system_prompt", "")
        for i in range(1, max_turns + 1):
            if self.stop_event.is_set():
                break
            self.run_turn(system_prompt, i, logger_obj)
        if feedback:
            fb = feedback.evaluate(self.state)
            print(fb)

    def request_stop(self) -> None:
        self.stop_event.set()
        self.barge_in_flag.set()
