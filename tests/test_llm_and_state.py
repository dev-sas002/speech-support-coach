from src.llm_module import LLMClient
from src.persona_loader import list_personas, load_persona
from src.state_manager import ConversationState


def test_conversation_state_as_messages_includes_system() -> None:
    state = ConversationState(session_id="s", persona_name="p", max_turns=2)
    state.add_turn("user", "hi")
    state.add_turn("assistant", "hello")

    msgs = state.as_messages("system prompt")
    assert msgs[0] == {"role": "system", "content": "system prompt"}
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant"]


def test_llm_client_complete_uses_injected_client(fake_groq_client) -> None:
    client = LLMClient(client=fake_groq_client)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    text, latency_ms, usage = client.complete(messages)

    assert text == "test completion"
    assert latency_ms >= 0
    # `usage is None or isinstance(usage, dict)` passed for literally any
    # return value the function could produce. complete() promises a plain
    # dict, so check that it is one and that it carries the counts.
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_persona_loader_can_list_and_load_personas(project_root) -> None:
    personas = list_personas(project_root)
    assert personas, "Expected at least one persona in config/personas"

    key, persona = personas[0]
    loaded = load_persona(key, project_root)
    assert loaded["name"] == persona["name"]
    assert "system_prompt" in loaded
