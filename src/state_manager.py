"""
The rolling conversation window.

Two bounds, not one. `max_turns` keeps the context recent; `max_prompt_chars`
keeps it *small*. They are different failure modes: eight turns of "yes" is
nothing, and eight turns of a customer reading out a statement is a prompt that
costs real money and real milliseconds on every subsequent turn.

Latency is the product here, and prompt length is the input the caller controls
and the system does not. An unbounded window means turn latency climbs through
a call — the longer someone talks to the agent, the slower it answers, which is
exactly backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Roughly 1,500 tokens at four characters per token, leaving comfortable room
#: for the system prompt and the reply inside a small model's context. Chosen
#: as a latency bound rather than a capacity one: this is about what can be
#: re-sent every turn without the pause becoming audible.
DEFAULT_MAX_PROMPT_CHARS = 6000


@dataclass
class ConversationState:
    """
    Track a single conversational session with the assistant.

    Stores the minimum needed to rebuild a chat-style message list, bounded by
    both a turn count and a character budget.
    """

    session_id: str
    persona_name: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    max_turns: int = 8
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS

    def add_turn(self, role: str, text: str) -> None:
        """Append a turn, then enforce both bounds oldest-first."""
        self.turns.append({"role": role, "text": text})
        self._enforce_bounds()

    def _enforce_bounds(self) -> None:
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

        # Always keep the most recent turn: dropping the thing that was just
        # said to stay under budget would leave the model answering the
        # previous question.
        while len(self.turns) > 1 and self.total_chars > self.max_prompt_chars:
            self.turns.pop(0)

    @property
    def total_chars(self) -> int:
        """Characters currently held in the window, excluding the system prompt."""
        return sum(len(t.get("text", "")) for t in self.turns)

    def as_messages(self, system_prompt: str) -> list[dict[str, str]]:
        """
        Convert the stored turns into an LLM-ready message list.

        The first message is always the system prompt, followed by the retained
        turns in the order they occurred.
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": t["role"], "content": t["text"]} for t in self.turns)
        return messages
