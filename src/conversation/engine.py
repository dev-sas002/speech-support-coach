"""
The turn loop, with no user interface attached.

This is the layer the Streamlit page and the CLI client both sit on. It owns
the one thing they used to duplicate: take what the customer said, fold it into
the conversation state, ask the language model, fold the answer back in, and
time every part of it.

Keeping it here rather than in the view matters for two reasons. The obvious
one is that the same behaviour was written twice and could drift. The less
obvious one is that latency is the product: a turn loop that lives inside a
render function can only be measured by rendering, so the numbers the project
exists to show were unmeasurable outside a browser. `ConversationEngine` is
plain Python and can be driven by a benchmark, a test, or a UI.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from ..llm_module import approx_tokens
from ..providers.base import LanguageModel, Message
from ..state_manager import ConversationState

#: Where a spoken reply stops feeling like a conversation. Human turn-taking
#: gaps sit around 200 ms; past roughly a second of silence a caller assumes
#: the line has dropped. Nothing fails when the budget is missed — the turn is
#: flagged, so a regression is visible instead of merely felt.
DEFAULT_TURN_BUDGET_MS = 1200.0


def turn_budget_ms() -> float:
    """The per-turn latency budget, overridable with TURN_BUDGET_MS."""
    raw = (os.getenv("TURN_BUDGET_MS") or "").strip()
    if not raw:
        return DEFAULT_TURN_BUDGET_MS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TURN_BUDGET_MS
    return value if value > 0 else DEFAULT_TURN_BUDGET_MS


@dataclass(slots=True)
class TurnMetrics:
    """
    What one turn cost, stage by stage.

    `first_token_ms` is the number that decides whether a voice agent feels
    alive: it is when the reply *starts*, not when it finishes. It is None for
    a non-streaming turn, where nothing is knowable before the whole response
    lands — which is itself the argument for streaming.
    """

    index: int
    llm_ms: float
    total_ms: float
    first_token_ms: float | None = None
    asr_ms: float | None = None
    tts_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    budget_ms: float = DEFAULT_TURN_BUDGET_MS
    streamed: bool = False

    @property
    def responsiveness_ms(self) -> float:
        """
        How long the caller waited before hearing anything.

        Time to first token when the reply was streamed, whole-turn latency
        when it was not. This is what the budget is measured against; total
        turn time is the wrong target, because a long answer that starts
        immediately feels fast and a short one that starts late does not.
        """
        if self.first_token_ms is not None:
            return self.first_token_ms
        return self.total_ms

    @property
    def over_budget(self) -> bool:
        return self.responsiveness_ms > self.budget_ms

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["responsiveness_ms"] = round(self.responsiveness_ms, 1)
        data["over_budget"] = self.over_budget
        return data


@dataclass(slots=True)
class Turn:
    """A completed exchange: what was said, what came back, what it cost."""

    user_text: str
    assistant_text: str
    metrics: TurnMetrics


@dataclass
class ConversationEngine:
    """
    One conversation with one persona, over one language-model provider.

    The provider is injected rather than constructed, which is what makes the
    engine testable without a network and swappable without an edit.
    """

    persona: dict[str, Any]
    llm: LanguageModel
    state: ConversationState = field(default=None)  # type: ignore[assignment]
    budget_ms: float = field(default_factory=turn_budget_ms)
    turns: list[Turn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = ConversationState(
                session_id=self.persona.get("scenario", "session"),
                persona_name=self.persona.get("name", "Assistant"),
            )

    # -- prompt assembly ---------------------------------------------------

    @property
    def system_prompt(self) -> str:
        return self.persona.get("system_prompt", "You are a helpful assistant.")

    def _messages_for(self, user_text: str) -> list[Message]:
        self.state.add_turn("user", user_text)
        return self.state.as_messages(self.system_prompt)

    def _record(
        self,
        user_text: str,
        assistant_text: str,
        metrics: TurnMetrics,
    ) -> Turn:
        self.state.add_turn("assistant", assistant_text)
        turn = Turn(user_text=user_text, assistant_text=assistant_text, metrics=metrics)
        self.turns.append(turn)
        return turn

    # -- the two ways to take a turn ---------------------------------------

    def respond(self, user_text: str, asr_ms: float | None = None) -> Turn:
        """One whole turn, waiting for the complete reply."""
        started = time.perf_counter()
        messages = self._messages_for(user_text)
        reply = self.llm.complete(messages)
        total_ms = (time.perf_counter() - started) * 1000 + (asr_ms or 0.0)

        usage = reply.usage or {}
        metrics = TurnMetrics(
            index=len(self.turns) + 1,
            llm_ms=reply.latency_ms,
            total_ms=total_ms,
            asr_ms=asr_ms,
            tokens_in=usage.get("prompt_tokens") or approx_tokens(user_text),
            tokens_out=usage.get("completion_tokens") or approx_tokens(reply.text),
            budget_ms=self.budget_ms,
            streamed=False,
        )
        return self._record(user_text, reply.text, metrics)

    def stream_respond(self, user_text: str, asr_ms: float | None = None) -> Iterator[str]:
        """
        One turn, yielding text as it arrives.

        The turn is only complete when the generator is exhausted, so the
        assistant message and `last_turn` land at that point. Callers that need
        the metrics read `engine.last_turn` after consuming the stream.
        """
        started = time.perf_counter()
        messages = self._messages_for(user_text)
        first_token_ms: float | None = None
        pieces: list[str] = []

        for piece in self.llm.stream(messages):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            pieces.append(piece)
            yield piece

        elapsed_ms = (time.perf_counter() - started) * 1000
        assistant_text = "".join(pieces).strip()
        metrics = TurnMetrics(
            index=len(self.turns) + 1,
            llm_ms=elapsed_ms,
            total_ms=elapsed_ms + (asr_ms or 0.0),
            first_token_ms=first_token_ms,
            asr_ms=asr_ms,
            tokens_in=approx_tokens(user_text),
            tokens_out=approx_tokens(assistant_text),
            budget_ms=self.budget_ms,
            streamed=True,
        )
        self._record(user_text, assistant_text, metrics)

    # -- inspection --------------------------------------------------------

    @property
    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def reset(self) -> None:
        self.state.turns = []
        self.turns = []

    def metrics_table(self) -> list[dict[str, Any]]:
        """Every turn's metrics, shaped for a dataframe or a CSV."""
        return [t.metrics.as_dict() for t in self.turns]

    def summary(self) -> dict[str, Any]:
        """Aggregate latency for the call so far."""
        if not self.turns:
            return {"turns": 0}
        responsive = sorted(t.metrics.responsiveness_ms for t in self.turns)
        return {
            "turns": len(self.turns),
            "mean_response_ms": sum(responsive) / len(responsive),
            "p95_response_ms": responsive[min(len(responsive) - 1, int(len(responsive) * 0.95))],
            "worst_response_ms": responsive[-1],
            "over_budget": sum(1 for t in self.turns if t.metrics.over_budget),
            "budget_ms": self.budget_ms,
        }
