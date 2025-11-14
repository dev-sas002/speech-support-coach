"""
Exporting a call, and seeding one so the app is not empty on first boot.
"""

from __future__ import annotations

import json

from src.conversation import ConversationEngine, export_json, export_markdown, seed_demo_call
from src.conversation.seed import DEMO_OPENINGS, FALLBACK_OPENING, openings_for
from src.providers.base import LLMReply

PERSONA = {
    "name": "Lost Card Support",
    "scenario": "Card Lost",
    "system_prompt": "You are a bank support agent.",
}


class FakeModel:
    name = "fake"

    def __init__(self, reply: str = "I can help with that.") -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return LLMReply(text=self.reply, latency_ms=8.0, usage={"total_tokens": 20})

    def stream(self, messages):
        self.calls += 1
        yield self.reply

    def status(self):  # pragma: no cover
        raise NotImplementedError


def populated_engine(turns: int = 2) -> ConversationEngine:
    engine = ConversationEngine(persona=PERSONA, llm=FakeModel())
    for index in range(turns):
        engine.respond(f"customer line {index}")
    return engine


class TestJsonExport:
    def test_it_carries_the_header_the_transcript_and_the_metrics(self) -> None:
        payload = json.loads(export_json(populated_engine()))

        assert payload["persona"] == "Lost Card Support"
        assert payload["provider"] == "fake"
        assert payload["summary"]["turns"] == 2
        assert [t["customer"] for t in payload["turns"]] == [
            "customer line 0",
            "customer line 1",
        ]
        assert payload["turns"][0]["metrics"]["responsiveness_ms"] >= 0

    def test_a_review_is_included_when_one_was_run(self) -> None:
        payload = json.loads(export_json(populated_engine(), review="**Score: 80/100**"))
        assert payload["review"].startswith("**Score")

    def test_an_empty_call_still_produces_valid_json(self) -> None:
        payload = json.loads(export_json(ConversationEngine(persona=PERSONA, llm=FakeModel())))
        assert payload["turns"] == []
        assert payload["summary"] == {"turns": 0}


class TestMarkdownExport:
    def test_both_sides_of_every_turn_appear(self) -> None:
        markdown = export_markdown(populated_engine())

        assert "# Support call — Lost Card Support" in markdown
        assert "**Customer:** customer line 0" in markdown
        assert "**Lost Card Support:** I can help with that." in markdown

    def test_the_latency_a_turn_was_produced_at_travels_with_it(self) -> None:
        # A transcript without the timings loses the half of this project that
        # is actually interesting.
        assert "Responded in" in export_markdown(populated_engine())

    def test_a_review_is_appended_when_one_was_run(self) -> None:
        markdown = export_markdown(populated_engine(), review="Looks good.")
        assert markdown.rstrip().endswith("Looks good.")

    def test_an_empty_call_exports_a_header_and_no_turns(self) -> None:
        markdown = export_markdown(ConversationEngine(persona=PERSONA, llm=FakeModel()))
        assert "## Transcript" in markdown
        assert "### Turn" not in markdown


class TestSeeding:
    def test_the_openings_match_the_shipped_scenarios(self) -> None:
        assert set(DEMO_OPENINGS) == {"card_lost", "transfer_failed", "account_locked"}

    def test_a_persona_is_matched_to_its_lines_regardless_of_casing(self) -> None:
        assert openings_for({"scenario": "Card Lost"}) == DEMO_OPENINGS["card_lost"]

    def test_an_unknown_scenario_gets_the_generic_opening(self) -> None:
        assert openings_for({"scenario": "Mortgage"}) == FALLBACK_OPENING

    def test_seeding_runs_the_lines_through_the_real_engine(self) -> None:
        # The replies are produced, not stored, so the seeded transcript has
        # gone through the same prompt assembly and timing as anything typed.
        model = FakeModel()
        engine = ConversationEngine(persona=PERSONA, llm=model)
        seed_demo_call(engine)

        assert model.calls == 3
        assert len(engine.turns) == 3
        assert all(t.metrics.total_ms >= 0 for t in engine.turns)

    def test_seeded_turns_are_streamed_so_they_carry_a_first_token_time(self) -> None:
        # A seeded call built on the whole-response path would show no time to
        # first token, which is the one number the page exists to show.
        engine = ConversationEngine(persona=PERSONA, llm=FakeModel())
        seed_demo_call(engine)

        assert all(t.metrics.streamed for t in engine.turns)
        assert all(t.metrics.first_token_ms is not None for t in engine.turns)

    def test_seeding_an_already_started_call_does_nothing(self) -> None:
        engine = populated_engine(1)
        seed_demo_call(engine)
        assert len(engine.turns) == 1

    def test_the_turn_count_is_respected(self) -> None:
        engine = ConversationEngine(persona=PERSONA, llm=FakeModel())
        seed_demo_call(engine, turns=2)
        assert len(engine.turns) == 2

    def test_a_provider_that_fails_leaves_an_empty_conversation(self) -> None:
        # A convenience must never become an error page.
        class Broken(FakeModel):
            def stream(self, messages):
                raise RuntimeError("no provider")

        engine = ConversationEngine(persona=PERSONA, llm=Broken())
        seed_demo_call(engine)
        assert engine.turns == []
