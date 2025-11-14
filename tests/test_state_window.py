"""
The rolling window's two bounds.

`max_turns` keeps the context recent; `max_prompt_chars` keeps it small. They
fail differently, which is why there are two of them: eight turns of "yes" is
nothing, and eight turns of a customer reading out a bank statement is a prompt
that is re-sent, and re-paid for, on every subsequent turn.
"""

from __future__ import annotations

import pytest

from src.state_manager import DEFAULT_MAX_PROMPT_CHARS, ConversationState


def state(**kwargs) -> ConversationState:
    kwargs.setdefault("session_id", "s")
    kwargs.setdefault("persona_name", "p")
    return ConversationState(**kwargs)


class TestTurnBound:
    def test_the_oldest_turns_fall_out_first(self) -> None:
        window = state(max_turns=3)
        for index in range(5):
            window.add_turn("user", f"msg {index}")

        assert [t["text"] for t in window.turns] == ["msg 2", "msg 3", "msg 4"]

    def test_the_default_keeps_a_few_exchanges(self) -> None:
        assert state().max_turns == 8


class TestCharacterBound:
    def test_a_long_conversation_stops_growing(self) -> None:
        window = state(max_turns=100, max_prompt_chars=300)
        for _ in range(20):
            window.add_turn("user", "x" * 100)

        assert window.total_chars <= 300

    def test_the_most_recent_turn_is_never_dropped(self) -> None:
        # Dropping what was just said to stay under budget would leave the
        # model answering the previous question.
        window = state(max_turns=100, max_prompt_chars=10)
        window.add_turn("user", "a" * 500)

        assert len(window.turns) == 1
        assert window.total_chars == 500

    def test_a_single_oversized_turn_evicts_the_history(self) -> None:
        window = state(max_turns=100, max_prompt_chars=50)
        window.add_turn("user", "short")
        window.add_turn("assistant", "also short")
        window.add_turn("user", "z" * 200)

        assert [t["text"] for t in window.turns] == ["z" * 200]

    def test_a_normal_conversation_is_untouched_by_the_bound(self) -> None:
        window = state()
        for index in range(4):
            window.add_turn("user", f"a sentence of ordinary length, number {index}")

        assert len(window.turns) == 4
        assert window.total_chars < DEFAULT_MAX_PROMPT_CHARS

    def test_the_default_budget_is_a_latency_bound_not_a_capacity_one(self) -> None:
        # Roughly 1,500 tokens: small enough that re-sending it every turn does
        # not become audible, not "as much as the model will take".
        assert DEFAULT_MAX_PROMPT_CHARS == 6000


class TestMessages:
    def test_the_system_prompt_always_leads(self) -> None:
        window = state()
        window.add_turn("user", "hello")
        messages = window.as_messages("be helpful")

        assert messages[0] == {"role": "system", "content": "be helpful"}
        assert messages[1] == {"role": "user", "content": "hello"}

    def test_only_the_retained_turns_are_sent(self) -> None:
        window = state(max_turns=2)
        for index in range(5):
            window.add_turn("user", f"msg {index}")

        assert len(window.as_messages("sys")) == 3

    @pytest.mark.parametrize("role", ["user", "assistant"])
    def test_roles_survive_the_conversion(self, role) -> None:
        window = state()
        window.add_turn(role, "text")
        assert window.as_messages("sys")[1]["role"] == role
