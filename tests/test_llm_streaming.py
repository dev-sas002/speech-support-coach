"""
Streaming chat completions, the offline stand-in behind them, and the
sentence splitter that turns a token stream into something speakable.

Every LLM call goes through an injected client; nothing reaches a network.
"""

from __future__ import annotations

import pytest

from src.llm_module import LLMClient, _usage_as_dict, request_timeout_s
from src.local_backend import DEFAULT_TIMEOUT_S, LOCAL_MODEL_NAME
from src.voice_client import split_sentences

MESSAGES = [
    {"role": "system", "content": "You are a bank support agent."},
    {"role": "user", "content": "I lost my card."},
]


class TestStreamChat:
    def test_the_tokens_arrive_in_order(self, fake_streaming_groq_client) -> None:
        client = LLMClient(client=fake_streaming_groq_client)
        stream, _ = client.stream_chat(MESSAGES)
        assert list(stream) == ["Hello", " there", "."]

    def test_a_trailing_none_delta_ends_the_stream_cleanly(
        self, fake_streaming_groq_client
    ) -> None:
        # The SDK signals completion with a None delta; it must not surface as
        # an empty token or an exception.
        client = LLMClient(client=fake_streaming_groq_client)
        stream, _ = client.stream_chat(MESSAGES)
        assert "" not in list(stream)

    def test_the_request_carries_the_configured_parameters(
        self, fake_streaming_groq_client
    ) -> None:
        client = LLMClient(client=fake_streaming_groq_client, temperature=0.9, max_tokens=64)
        client.stream_chat(MESSAGES)

        sent = fake_streaming_groq_client.chat.completions.calls[0]
        assert sent["stream"] is True
        assert sent["messages"] == MESSAGES
        assert sent["temperature"] == 0.9
        assert sent["max_tokens"] == 64

    def test_time_to_first_token_is_unknown_until_the_stream_runs(
        self, fake_streaming_groq_client
    ) -> None:
        # The tuple is built before the caller consumes a token, so the second
        # element is the time to open the stream and nothing more. The real
        # first-token latency was previously computed into a variable that was
        # always still None at return time.
        client = LLMClient(client=fake_streaming_groq_client)
        stream, open_ms = client.stream_chat(MESSAGES)

        assert open_ms >= 0
        assert client.last_first_token_ms is None

        next(iter(stream))
        assert client.last_first_token_ms is not None
        assert client.last_first_token_ms >= 0

    def test_a_stream_that_yields_nothing_leaves_the_latency_unset(self) -> None:
        class _Empty:
            def __init__(self) -> None:
                self.chat = type(
                    "C", (), {"completions": type("D", (), {"create": lambda s, **k: iter(())})()}
                )()

        client = LLMClient(client=_Empty())
        stream, _ = client.stream_chat(MESSAGES)
        assert list(stream) == []
        assert client.last_first_token_ms is None

    def test_a_provider_failure_becomes_a_named_error(self) -> None:
        class _Boom:
            def __init__(self) -> None:
                def create(**kwargs):
                    raise ConnectionError("network down")

                self.chat = type(
                    "C", (), {"completions": type("D", (), {"create": staticmethod(create)})()}
                )()

        with pytest.raises(RuntimeError, match="Failed to start streaming"):
            LLMClient(client=_Boom()).stream_chat(MESSAGES)

    def test_a_malformed_chunk_is_skipped_rather_than_raising(self) -> None:
        class _Ragged:
            def __init__(self) -> None:
                def create(**kwargs):
                    yield object()  # no .choices at all
                    yield type(
                        "Chunk",
                        (),
                        {
                            "choices": [
                                type("Ch", (), {"delta": type("D", (), {"content": "ok"})()})()
                            ]
                        },
                    )()

                self.chat = type(
                    "C", (), {"completions": type("D", (), {"create": staticmethod(create)})()}
                )()

        stream, _ = LLMClient(client=_Ragged()).stream_chat(MESSAGES)
        assert list(stream) == ["ok"]


class TestOfflineStreaming:
    def test_the_offline_client_streams_its_scripted_line(self, monkeypatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        client = LLMClient()
        assert client.is_local is True
        assert client.config.model == LOCAL_MODEL_NAME

        stream, _ = client.stream_chat(MESSAGES)
        text = "".join(stream)
        assert "sorry" in text.lower()

    def test_the_offline_reply_is_identical_streamed_and_complete(self, monkeypatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        client = LLMClient()
        streamed = "".join(client.stream_chat(MESSAGES)[0])
        whole, _, _ = client.complete(MESSAGES)
        assert streamed == whole

    def test_offline_usage_is_reported_as_a_plain_dict(self, monkeypatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        _, _, usage = LLMClient().complete(MESSAGES)
        assert isinstance(usage, dict)
        assert usage["completion_tokens"] > 0


class TestUsageNormalisation:
    def test_none_stays_none(self) -> None:
        assert _usage_as_dict(None) is None

    def test_a_dict_passes_through(self) -> None:
        assert _usage_as_dict({"total_tokens": 7}) == {"total_tokens": 7}

    def test_a_pydantic_style_model_is_converted(self) -> None:
        class _Model:
            def model_dump(self):
                return {"total_tokens": 7}

        assert _usage_as_dict(_Model()) == {"total_tokens": 7}

    def test_a_failing_converter_falls_back_to_the_documented_fields(self) -> None:
        class _Awkward:
            prompt_tokens = 1
            completion_tokens = 2
            total_tokens = 3

            def model_dump(self):
                raise TypeError("not serialisable")

        assert _usage_as_dict(_Awkward()) == {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }

    def test_an_object_with_nothing_useful_yields_none(self) -> None:
        assert _usage_as_dict(object()) is None


class TestRequestTimeout:
    def test_the_default_is_used_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("GROQ_TIMEOUT_SECONDS", raising=False)
        assert request_timeout_s() == DEFAULT_TIMEOUT_S

    @pytest.mark.parametrize("value", ["", "   ", "not-a-number", "0", "-5"])
    def test_unusable_values_fall_back_rather_than_disabling_the_timeout(
        self, monkeypatch, value
    ) -> None:
        monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", value)
        assert request_timeout_s() == DEFAULT_TIMEOUT_S

    def test_a_valid_override_is_honoured(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "5.5")
        assert request_timeout_s() == 5.5


class TestSeedConfiguration:
    def test_the_seed_comes_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("SEED", "42")
        assert LLMClient(client=object()).config.seed == 42

    @pytest.mark.parametrize("value", ["", "   ", "not-a-number"])
    def test_an_unusable_seed_is_treated_as_unset(self, monkeypatch, value) -> None:
        monkeypatch.setenv("SEED", value)
        assert LLMClient(client=object()).config.seed is None


class TestSplitSentences:
    def test_tokens_are_regrouped_into_sentences(self) -> None:
        tokens = ["Hello", " there", ". ", "How ", "are ", "you", "? "]
        assert list(split_sentences(tokens)) == ["Hello there.", "How are you?"]

    def test_a_trailing_fragment_is_still_emitted(self) -> None:
        # Otherwise the last thing the assistant said would never be spoken.
        assert list(split_sentences(["no ", "final ", "punctuation"])) == ["no final punctuation"]

    def test_nothing_is_emitted_for_an_empty_stream(self) -> None:
        assert list(split_sentences([])) == []

    def test_whitespace_only_output_yields_nothing(self) -> None:
        assert list(split_sentences(["   ", "\n"])) == []

    def test_a_stop_flag_truncates_the_stream(self) -> None:
        assert list(split_sentences(["One. ", "Two. "], stop_flag=lambda: True)) == []

    def test_a_stop_suppresses_the_trailing_fragment(self) -> None:
        calls = {"n": 0}

        def stop_after_first() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        assert list(split_sentences(["One. ", "tail"], stop_flag=stop_after_first)) == ["One."]

    def test_every_token_is_reported_to_the_partial_callback(self) -> None:
        seen: list[str] = []
        list(split_sentences(["a", "b. "], on_partial=seen.append))
        assert seen == ["a", "b. "]

    def test_several_sentences_inside_one_token_are_split(self) -> None:
        assert list(split_sentences(["One. Two! Three? "])) == [
            "One.",
            "Two!",
            "Three?",
        ]
