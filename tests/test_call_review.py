"""
Post-call review: the deterministic compliance scan and the model-backed grade.

No test here reaches a model. The qualitative path is exercised with fakes that
return the JSON a real one would, the malformed output a real one sometimes
does, and the exception a real one occasionally throws.
"""

from __future__ import annotations

import json

import pytest

from src.analysis import review_call, scan_compliance
from src.local_backend import CLOSING, SCRIPTS
from src.providers.base import LLMReply
from src.state_manager import ConversationState


def conversation(*turns: tuple[str, str]) -> ConversationState:
    state = ConversationState(session_id="s", persona_name="Agent", max_turns=100)
    for role, text in turns:
        state.add_turn(role, text)
    return state


class ReviewingModel:
    """A model that returns whatever review payload the test gives it."""

    name = "fake-hosted"

    def __init__(self, payload, wrap: str = "{body}") -> None:
        self.payload = payload
        self.wrap = wrap
        self.seen: list[list[dict[str, str]]] = []

    def complete(self, messages):
        self.seen.append(messages)
        body = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return LLMReply(text=self.wrap.format(body=body), latency_ms=30.0)

    def stream(self, messages):  # pragma: no cover - review never streams
        raise NotImplementedError

    def status(self):  # pragma: no cover
        raise NotImplementedError


GOOD_PAYLOAD = {
    "score": 88,
    "summary": "Verified the caller before acting and explained the next steps.",
    "points": [
        {"label": "Verified identity first", "met": True, "note": "asked for DOB"},
        {"label": "Explained the hold", "met": True, "note": ""},
        {"label": "Confirmed a callback", "met": False, "note": "never offered"},
    ],
}


class TestComplianceScan:
    def test_the_shipped_scripts_break_no_rules(self) -> None:
        # A scanner that fires on the project's own correct dialogue would be
        # worse than no scanner at all, so this pins it against every line the
        # offline agent can say.
        state = ConversationState(session_id="s", persona_name="a", max_turns=999)
        for lines in SCRIPTS.values():
            for line in lines:
                state.add_turn("assistant", line)
        state.add_turn("assistant", CLOSING)

        assert scan_compliance(state) == []

    @pytest.mark.parametrize(
        ("utterance", "rule"),
        [
            ("Can you read out the full card number for me?", "full_card_number"),
            ("I just need you to confirm your PIN.", "pin"),
            ("What is your password, please?", "password"),
            ("Tell me the security code on the back.", "cvv"),
            ("I can guarantee a refund within 48 hours.", "guaranteed_refund"),
        ],
    )
    def test_a_forbidden_request_is_caught(self, utterance, rule) -> None:
        flags = scan_compliance(conversation(("assistant", utterance)))
        assert [f.rule for f in flags] == [rule]

    def test_normal_procedure_is_not_a_breach(self) -> None:
        state = conversation(
            ("assistant", "You'll be prompted to set a new password once it unlocks."),
            ("assistant", "Can you confirm the last four digits of your card?"),
            ("assistant", "I've started a password reset for you."),
        )
        assert scan_compliance(state) == []

    def test_only_the_agent_is_judged(self) -> None:
        # A customer volunteering their PIN is a different problem, and not an
        # agent compliance breach.
        state = conversation(("user", "My PIN is 4412, does that help?"))
        assert scan_compliance(state) == []

    def test_the_flag_quotes_the_words_that_triggered_it(self) -> None:
        flags = scan_compliance(conversation(("assistant", "For security, confirm your PIN now.")))
        assert "PIN" in flags[0].quote


class TestModelBackedReview:
    def test_a_well_formed_review_is_used(self) -> None:
        review = review_call(
            conversation(("user", "hello"), ("assistant", "Good morning.")),
            llm=ReviewingModel(GOOD_PAYLOAD),
        )

        assert review.source == "model"
        assert review.score == 88
        assert next(p.label for p in review.points) == "Verified identity first"
        assert review.points[2].met is False

    def test_the_transcript_reaches_the_model(self) -> None:
        model = ReviewingModel(GOOD_PAYLOAD)
        review_call(
            conversation(("user", "I lost my card"), ("assistant", "Sorry to hear.")),
            llm=model,
            persona_name="Lost Card Support",
        )

        sent = model.seen[0][-1]["content"]
        assert "Customer: I lost my card" in sent
        assert "Lost Card Support: Sorry to hear." in sent

    def test_prose_around_the_json_is_tolerated(self) -> None:
        model = ReviewingModel(GOOD_PAYLOAD, wrap="Here is my review:\n{body}\nHope that helps.")
        assert review_call(conversation(("user", "hi")), llm=model).source == "model"

    def test_a_score_outside_the_range_is_clamped(self) -> None:
        model = ReviewingModel({**GOOD_PAYLOAD, "score": 140})
        assert review_call(conversation(("user", "hi")), llm=model).score == 100

    @pytest.mark.parametrize(
        "payload",
        ["not json at all", "{", json.dumps({"score": 70}), json.dumps({"points": []})],
    )
    def test_an_unusable_reply_falls_back_to_the_heuristic(self, payload) -> None:
        review = review_call(conversation(("user", "hi")), llm=ReviewingModel(payload))
        assert review.source == "heuristic"

    def test_a_model_that_throws_does_not_break_the_review(self) -> None:
        class Broken(ReviewingModel):
            def complete(self, messages):
                raise RuntimeError("provider is down")

        review = review_call(conversation(("user", "hi")), llm=Broken(None))
        assert review.source == "heuristic"

    def test_the_scripted_agent_is_never_asked_to_grade(self) -> None:
        # It replies with support dialogue, not with an assessment; asking it
        # to score a call would produce confident nonsense.
        class Scripted(ReviewingModel):
            name = "scripted"

        model = Scripted(GOOD_PAYLOAD)
        review = review_call(conversation(("user", "hi")), llm=model)

        assert review.source == "heuristic"
        assert model.seen == []


class TestHeuristicFallback:
    def test_no_model_means_the_keyword_review(self) -> None:
        review = review_call(
            conversation(
                ("user", "Hello, I've lost my card"),
                ("assistant", "I'm sorry to hear that. Can you confirm your name?"),
                ("user", "Sure"),
                ("assistant", "I will block the card now."),
            )
        )

        assert review.source == "heuristic"
        assert review.score == 100
        assert all(point.met for point in review.points)

    def test_a_call_that_missed_everything_scores_zero(self) -> None:
        review = review_call(conversation(("user", "..."), ("assistant", "Mm.")))
        assert review.score == 0
        assert not any(point.met for point in review.points)

    def test_an_empty_conversation_is_reported_as_such(self) -> None:
        review = review_call(ConversationState(session_id="s", persona_name="a"))
        assert review.score == 0
        assert review.points == []
        assert "nothing to review" in review.summary


class TestScoreAndRendering:
    def test_a_compliance_breach_caps_the_score(self) -> None:
        # However well the model thought the call went, asking for a PIN is not
        # a matter of opinion.
        state = conversation(
            ("user", "Hello"),
            ("assistant", "Good morning. Please confirm your PIN."),
        )
        review = review_call(state, llm=ReviewingModel(GOOD_PAYLOAD))

        assert review.score == 50
        assert review.passed_compliance is False

    def test_the_markdown_names_where_the_review_came_from(self) -> None:
        model_review = review_call(conversation(("user", "hi")), llm=ReviewingModel(GOOD_PAYLOAD))
        heuristic_review = review_call(conversation(("user", "hi")))

        assert "language model" in model_review.to_markdown()
        assert "keyword heuristic" in heuristic_review.to_markdown()

    def test_a_clean_call_says_so(self) -> None:
        markdown = review_call(conversation(("user", "hi"))).to_markdown()
        assert "No compliance rules were broken" in markdown

    def test_flags_are_rendered_with_their_quotes(self) -> None:
        markdown = review_call(
            conversation(("assistant", "Please confirm your PIN."))
        ).to_markdown()
        assert "Compliance flags" in markdown
        assert "PIN" in markdown

    def test_the_dict_form_round_trips_through_json(self) -> None:
        review = review_call(
            conversation(("assistant", "Please confirm your PIN.")),
            llm=ReviewingModel(GOOD_PAYLOAD),
        )
        assert json.loads(json.dumps(review.as_dict()))["flags"][0]["rule"] == "pin"
