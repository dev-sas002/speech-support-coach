import sys
import types
from collections.abc import Iterable
from pathlib import Path

import pytest


def _install_fake_groq_module() -> None:
    """
    Install a lightweight fake `groq` module into sys.modules if it is missing.

    This keeps imports of `src.llm_module` working in test environments
    where the real Groq SDK is not installed, without affecting runtime code.
    """
    if "groq" in sys.modules:
        return

    fake_groq = types.ModuleType("groq")

    class FakeGroqClient:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - trivial
            pass

    fake_groq.Groq = FakeGroqClient
    sys.modules["groq"] = fake_groq


def _install_fake_webrtcvad_module() -> None:
    """
    Install a lightweight fake `webrtcvad` module if the real one is missing.

    `src.asr_module` imports it at module scope, and it is a C extension with
    no wheel for every interpreter. Stubbing it keeps the conversation modules
    importable in a test environment; every test that cares about voice
    activity supplies its own detector, so nothing here depends on the stub's
    behaviour.
    """
    if "webrtcvad" in sys.modules:
        return
    try:
        import webrtcvad  # noqa: F401

        return
    except ImportError:
        pass

    fake = types.ModuleType("webrtcvad")

    class FakeVad:
        def __init__(self, aggressiveness: int = 0) -> None:
            self.aggressiveness = aggressiveness

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            return False

    fake.Vad = FakeVad
    sys.modules["webrtcvad"] = fake


_install_fake_groq_module()
_install_fake_webrtcvad_module()


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch) -> None:
    """
    Strip provider credentials from the environment for every test.

    A developer with a working GROQ_API_KEY exported would otherwise have any
    client built without an injected stand-in silently construct a real Groq
    client and reach the network. Tests that care about key handling set the
    variable themselves; this fixture runs first, so they still win.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


@pytest.fixture
def personas_dir() -> Path:
    """
    Fixture pointing at the real personas directory in the project.

    Tests that exercise persona loading can use this to avoid hard-coded paths.
    """
    return Path(__file__).resolve().parent.parent / "config" / "personas"


class _FakeGroqUsage:
    """Usage block, matching the attributes LLMClient reads."""

    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


class _FakeGroqMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


class _FakeGroqChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeGroqMessage(content)


class _FakeGroqResponse:
    """
    The response shape LLMClient expects.

    This class was referenced by the fixture below and defined nowhere, so
    `test_llm_client_complete_uses_injected_client` raised NameError inside the
    fixture and had never passed.
    """

    def __init__(self, content: str) -> None:
        self.choices = [_FakeGroqChoice(content)]
        self.usage = _FakeGroqUsage()
        self.model = "fake-model"


class _FakeGroqChatCompletions:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def create(self, **kwargs) -> "_FakeGroqResponse":
        # Minimal response object with the shape used by LLMClient.
        return _FakeGroqResponse(self._response_text)


class _FakeGroqChat:
    def __init__(self, response_text: str) -> None:
        self.completions = _FakeGroqChatCompletions(response_text)


class _FakeGroqClient:
    def __init__(self, response_text: str = "hello") -> None:
        self.chat = _FakeGroqChat(response_text)


class _FakeChunkChoiceDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChunkChoice:
    def __init__(self, content: str) -> None:
        self.delta = _FakeChunkChoiceDelta(content)


class _FakeStreamChunk:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChunkChoice(content)]


class _StreamingCompletions:
    def __init__(self, tokens: Iterable[str]) -> None:
        self._tokens: list[str] = list(tokens)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        def chunks():
            for tok in self._tokens:
                yield _FakeStreamChunk(tok)
            # The SDK signals completion with a None delta; the caller relies
            # on the generator simply ending, so the fake emits both.
            yield _FakeStreamChunk(None)

        return chunks()


class _StreamingChat:
    def __init__(self, tokens: Iterable[str]) -> None:
        self.completions = _StreamingCompletions(tokens)


class _FakeGroqStreamingClient(_FakeGroqClient):
    """
    A fake client whose `chat.completions.create` streams the given tokens.

    `__post_init__` is a dataclass hook and this is a plain class, so it was
    never called: `self.chat` stayed the non-streaming parent's, the nested
    streaming classes were unreachable, and no test could have used this to
    exercise streaming at all.
    """

    def __init__(self, tokens: Iterable[str]) -> None:
        super().__init__()
        self.chat = _StreamingChat(tokens)


@pytest.fixture
def fake_groq_client() -> _FakeGroqClient:
    """Provide a fake Groq client that returns a fixed completion."""
    return _FakeGroqClient("test completion")


@pytest.fixture
def fake_streaming_groq_client() -> _FakeGroqStreamingClient:
    """A fake Groq client that streams a fixed sequence of token deltas."""
    return _FakeGroqStreamingClient(["Hello", " there", "."])


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent
