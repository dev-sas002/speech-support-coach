"""
The turn loop: what it records, what it sends, and what it refuses to grow.

Every test here injects a fake language model, so nothing reaches a network and
the timings are the engine's own overhead rather than a provider's.
"""

from __future__ import annotations

import time

import pytest

from src.conversation import ConversationEngine, TurnMetrics, turn_budget_ms
from src.conversation.engine import DEFAULT_TURN_BUDGET_MS
from src.providers.base import LLMReply
from src.state_manager import ConversationState

PERSONA = {
    "name": "Lost Card Support",
    "scenario": "Card Lost",
    "system_prompt": "You are a bank support agent.",
}


class FakeModel:
    """A language model that answers instantly and remembers what it was asked."""

    name = "fake"

    def __init__(self, reply: str = "Certainly. I can help with that.") -> None:
        self.reply = reply
        self.seen: list[list[dict[str, str]]] = []

    def complete(self, messages):
        self.seen.append(messages)
        return LLMReply(text=self.reply, latency_ms=12.0, usage={"total_tokens": 42})

    def stream(self, messages):
        self.seen.append(messages)
        for word in self.reply.split(" "):
            yield word + " "

    def status(self):  # pragma: no cover - not exercised by the engine
        raise NotImplementedError


class SlowModel(FakeModel):
    """Answers, but not quickly. Used to check the budget actually bites."""

    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s

    def complete(self, messages):
        time.sleep(self.delay_s)
        return super().complete(messages)


def engine_for(model: FakeModel, **kwargs) -> ConversationEngine:
    return ConversationEngine(persona=PERSONA, llm=model, **kwargs)


class TestWholeResponseTurns:
    def test_a_turn_records_both_sides_of_the_exchange(self) -> None:
        engine = engine_for(FakeModel("Of course."))
        turn = engine.respond("I lost my card.")

        assert turn.user_text == "I lost my card."
        assert turn.assistant_text == "Of course."
        assert [t["role"] for t in engine.state.turns] == ["user", "assistant"]

    def test_the_persona_leads_the_prompt(self) -> None:
        model = FakeModel()
        engine_for(model).respond("Hello.")

        assert model.seen[0][0] == {
            "role": "system",
            "content": PERSONA["system_prompt"],
        }

    def test_usage_from_the_provider_is_preferred_over_the_estimate(self) -> None:
        class Detailed(FakeModel):
            def complete(self, messages):
                return LLMReply(
                    text="ok",
                    latency_ms=5.0,
                    usage={"prompt_tokens": 111, "completion_tokens": 7},
                )

        turn = engine_for(Detailed()).respond("hello")
        assert (turn.metrics.tokens_in, turn.metrics.tokens_out) == (111, 7)

    def test_a_non_streaming_turn_has_no_first_token_time(self) -> None:
        # Nothing is knowable before the whole response lands, and claiming a
        # number here would be the dishonest kind of metric.
        turn = engine_for(FakeModel()).respond("hello")
        assert turn.metrics.first_token_ms is None
        assert turn.metrics.streamed is False

    def test_transcription_time_is_folded_into_the_turn(self) -> None:
        turn = engine_for(FakeModel()).respond("hello", asr_ms=250.0)
        assert turn.metrics.asr_ms == 250.0
        assert turn.metrics.total_ms >= 250.0


class TestStreamingTurns:
    def test_the_caller_sees_pieces_before_the_turn_is_recorded(self) -> None:
        engine = engine_for(FakeModel("One two three."))
        stream = engine.stream_respond("hello")

        first = next(stream)
        assert first.strip() == "One"
        # The assistant turn only exists once the stream has finished.
        assert engine.turns == []

        list(stream)
        assert engine.last_turn.assistant_text == "One two three."

    def test_a_streamed_turn_measures_time_to_first_token(self) -> None:
        engine = engine_for(FakeModel())
        list(engine.stream_respond("hello"))

        metrics = engine.last_turn.metrics
        assert metrics.streamed is True
        assert metrics.first_token_ms is not None
        assert metrics.first_token_ms <= metrics.llm_ms

    def test_responsiveness_is_the_first_token_when_streaming(self) -> None:
        engine = engine_for(FakeModel())
        list(engine.stream_respond("hello"))
        metrics = engine.last_turn.metrics

        assert metrics.responsiveness_ms == metrics.first_token_ms

    def test_an_empty_stream_still_closes_the_turn(self) -> None:
        class Silent(FakeModel):
            def stream(self, messages):
                return iter(())

        engine = engine_for(Silent())
        assert list(engine.stream_respond("hello")) == []
        assert engine.last_turn.assistant_text == ""
        assert engine.last_turn.metrics.first_token_ms is None


class TestBudget:
    def test_a_fast_turn_is_within_budget(self) -> None:
        turn = engine_for(FakeModel()).respond("hello")
        assert turn.metrics.over_budget is False

    def test_a_slow_turn_is_flagged_rather_than_failed(self) -> None:
        # Nothing raises: a voice agent that throws away a slow answer is worse
        # than one that delivers it late. The turn is marked so the regression
        # is visible.
        engine = engine_for(SlowModel(0.05), budget_ms=10.0)
        turn = engine.respond("hello")

        assert turn.metrics.over_budget is True
        assert turn.assistant_text

    def test_the_budget_comes_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("TURN_BUDGET_MS", "450")
        assert turn_budget_ms() == 450.0

    @pytest.mark.parametrize("value", ["", "   ", "soon", "0", "-1"])
    def test_an_unusable_budget_falls_back_to_the_default(self, monkeypatch, value) -> None:
        monkeypatch.setenv("TURN_BUDGET_MS", value)
        assert turn_budget_ms() == DEFAULT_TURN_BUDGET_MS


class TestWindow:
    def test_the_prompt_does_not_grow_without_bound(self) -> None:
        # Latency is the product here, so an unbounded window means the agent
        # gets slower the longer someone talks to it — exactly backwards.
        engine = engine_for(
            FakeModel("short."),
            state=ConversationState(
                session_id="s", persona_name="p", max_turns=100, max_prompt_chars=400
            ),
        )
        for _ in range(20):
            engine.respond("x" * 150)

        assert engine.state.total_chars <= 400
        assert len(engine.turns) == 20  # the transcript keeps everything

    def test_the_transcript_outlives_the_window(self) -> None:
        engine = engine_for(
            FakeModel(),
            state=ConversationState(
                session_id="s", persona_name="p", max_turns=2, max_prompt_chars=10_000
            ),
        )
        for index in range(4):
            engine.respond(f"message {index}")

        assert len(engine.state.turns) == 2
        assert [t.user_text for t in engine.turns] == [f"message {i}" for i in range(4)]


class TestSummary:
    def test_an_untouched_conversation_summarises_to_nothing(self) -> None:
        assert engine_for(FakeModel()).summary() == {"turns": 0}

    def test_the_summary_counts_the_turns_that_missed_the_budget(self) -> None:
        engine = engine_for(SlowModel(0.03), budget_ms=5.0)
        engine.respond("one")
        engine.respond("two")

        summary = engine.summary()
        assert summary["turns"] == 2
        assert summary["over_budget"] == 2
        assert summary["worst_response_ms"] >= summary["mean_response_ms"]

    def test_reset_clears_both_the_window_and_the_transcript(self) -> None:
        engine = engine_for(FakeModel())
        engine.respond("hello")
        engine.reset()

        assert engine.turns == []
        assert engine.state.turns == []

    def test_the_metrics_table_is_one_row_per_turn(self) -> None:
        engine = engine_for(FakeModel())
        engine.respond("one")
        engine.respond("two")

        rows = engine.metrics_table()
        assert [row["index"] for row in rows] == [1, 2]
        assert all("responsiveness_ms" in row for row in rows)


def test_metrics_without_a_stream_fall_back_to_whole_turn_latency() -> None:
    metrics = TurnMetrics(index=1, llm_ms=900.0, total_ms=1000.0, budget_ms=500.0)
    assert metrics.responsiveness_ms == 1000.0
    assert metrics.over_budget is True
