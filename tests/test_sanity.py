from src.llm_module import approx_tokens
from src.state_manager import ConversationState


def test_approx_tokens_minimum_one() -> None:
    assert approx_tokens("") == 1
    assert approx_tokens("a") == 1


def test_approx_tokens_grows_with_text_length() -> None:
    # `>=` held even if the estimate were a constant, which is the one
    # behaviour this test exists to rule out.
    assert approx_tokens("abcd" * 10) > approx_tokens("abcd")
    assert approx_tokens("a" * 400) == 100


def test_conversation_state_turn_limit_and_messages() -> None:
    state = ConversationState(session_id="s", persona_name="p", max_turns=3)
    for i in range(5):
        state.add_turn("user", f"msg {i}")

    assert len(state.turns) == 3
    assert [t["text"] for t in state.turns] == ["msg 2", "msg 3", "msg 4"]

    msgs = state.as_messages(system_prompt="sys")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "sys"
    # System message + one per stored turn
    assert len(msgs) == 1 + len(state.turns)
