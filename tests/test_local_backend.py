"""
Tests for the offline backend and provider selection.

A bank-support training tool must not invent procedure. The offline agent is
scripted precisely so that a trainee is never shown fabricated financial advice
they cannot distinguish from the real thing, and several of these tests exist
to keep it that way.
"""

from __future__ import annotations

import pytest

from src.local_backend import (
    CLOSING,
    LOCAL_MODEL_NAME,
    SCRIPTS,
    LocalGroqClient,
    count_assistant_turns,
    detect_scenario,
    scripted_reply,
    usable_api_key,
)

SYSTEM = {
    "card_lost": "You are a bank support agent helping a customer who has lost their debit card.",
    "transfer_failed": "You are helping a customer whose transfer failed to complete.",
    "account_locked": "You are helping a customer whose account is locked after failed sign-ins.",
}


def convo(scenario: str, assistant_turns: int = 0):
    messages = [{"role": "system", "content": SYSTEM[scenario]}]
    for _ in range(assistant_turns):
        messages.append({"role": "assistant", "content": "..."})
        messages.append({"role": "user", "content": "ok"})
    return messages


class TestScenarioDetection:
    @pytest.mark.parametrize("scenario", list(SYSTEM))
    def test_the_persona_prompt_selects_its_script(self, scenario):
        assert detect_scenario(convo(scenario)) == scenario

    def test_an_unrecognised_prompt_falls_back_rather_than_failing(self):
        assert detect_scenario([{"role": "system", "content": "something else"}]) in SCRIPTS


class TestConversationProgress:
    def test_the_call_advances_with_each_turn(self):
        replies = [scripted_reply(convo("card_lost", n)) for n in range(4)]
        assert len(set(replies)) == 4, "the agent repeated itself"

    def test_the_first_turn_acknowledges_and_verifies_identity(self):
        first = scripted_reply(convo("card_lost", 0)).lower()
        assert "sorry" in first
        # Verifying identity before acting is the part of the procedure that
        # matters most in this scenario.
        assert "confirm" in first

    def test_the_script_ends_in_a_closing_rather_than_looping(self):
        beyond = scripted_reply(convo("card_lost", len(SCRIPTS["card_lost"]) + 3))
        assert beyond == CLOSING

    def test_each_scenario_has_its_own_wording(self):
        card = scripted_reply(convo("card_lost", 1))
        transfer = scripted_reply(convo("transfer_failed", 1))
        assert card != transfer

    def test_turn_counting_ignores_user_messages(self):
        assert count_assistant_turns(convo("card_lost", 3)) == 3


class TestDeterminism:
    def test_the_same_conversation_produces_the_same_reply(self):
        # A training tool whose answers move between runs cannot be reviewed.
        assert scripted_reply(convo("card_lost", 2)) == scripted_reply(convo("card_lost", 2))


class TestGroqClientShape:
    """The stand-in has to match the SDK's shape, since the callers are unchanged."""

    def test_non_streaming_completion_has_the_expected_attributes(self):
        client = LocalGroqClient()
        result = client.chat.completions.create(messages=convo("card_lost"))
        assert result.choices[0].message.content
        assert result.usage.completion_tokens > 0
        assert result.usage.prompt_tokens > 0

    def test_the_token_total_matches_its_parts(self):
        # total_tokens kept the dataclass default of 0 while the two components
        # were populated, so the UI's token column read zero every offline turn.
        usage = LocalGroqClient().chat.completions.create(messages=convo("card_lost")).usage
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_streaming_yields_delta_chunks(self):
        client = LocalGroqClient()
        stream = client.chat.completions.create(messages=convo("card_lost"), stream=True)
        text = "".join(c.choices[0].delta.content or "" for c in stream)
        assert text == scripted_reply(convo("card_lost"))

    def test_the_stream_ends_with_a_none_delta(self):
        # The SDK signals completion this way and the caller checks for it.
        chunks = list(
            LocalGroqClient().chat.completions.create(messages=convo("card_lost"), stream=True)
        )
        assert chunks[-1].choices[0].delta.content is None

    def test_transcription_admits_it_cannot_hear(self):
        # Returning invented words would be worse than returning nothing: the
        # rest of the pipeline would treat them as what the trainee said.
        text = LocalGroqClient().audio.transcriptions.create().text
        assert "unavailable" in text.lower()


class TestKeySelection:
    def test_no_key_is_no_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert usable_api_key() is None

    @pytest.mark.parametrize(
        "value",
        ["your_groq_api_key_here", "YOUR-KEY", "changeme", "<paste key>", "   "],
    )
    def test_placeholders_are_treated_as_absent(self, monkeypatch, value):
        monkeypatch.setenv("GROQ_API_KEY", value)
        assert usable_api_key() is None

    def test_a_real_looking_key_is_returned(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_abc123realkeyvalue")
        assert usable_api_key() == "gsk_abc123realkeyvalue"


class TestModelDefault:
    """
    The hosted default was "penai/gpt-oss-20b" — a missing leading "o" — while
    the README documented "openai/gpt-oss-20b". Anyone relying on the default
    got a model-not-found error from the vendor.

    Checked behaviourally rather than by grepping the source, so the test
    describes what the client does rather than how the file is written.
    """

    def test_the_hosted_default_model_is_spelled_correctly(self, monkeypatch):
        from src.llm_module import LLMClient

        monkeypatch.delenv("GROQ_LLM_MODEL", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_looks_like_a_real_key")

        client = LLMClient(client=object())
        assert client.config.model == "openai/gpt-oss-20b"

    def test_without_a_key_the_local_model_name_is_reported(self, monkeypatch):
        from src.llm_module import LLMClient

        monkeypatch.delenv("GROQ_LLM_MODEL", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        assert LLMClient().config.model == LOCAL_MODEL_NAME
