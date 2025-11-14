import csv

import pytest

from src.feedback import evaluate
from src.logger import LatencyLogger
from src.state_manager import ConversationState


def test_latency_logger_writes_header_and_row(tmp_path) -> None:
    path = tmp_path / "latency.csv"
    logger = LatencyLogger(str(path))

    logger.log_turn(
        turn=1,
        persona="Test",
        asr_ms=10.0,
        llm_ms=20.0,
        tts_ms=30.0,
        total_ms=60.0,
        input_chars=5,
        output_chars=10,
        llm_tokens_in=3,
        llm_tokens_out=4,
        asr_secs=1.2,
        tts_chars=10,
        cost_est_usd=0.001,
        error=None,
    )

    rows = list(csv.DictReader(path.read_text().splitlines()))
    assert len(rows) == 1

    row = rows[0]
    # The header only told us the columns exist; these check the values
    # actually land in them.
    assert row["turn"] == "1"
    assert row["persona"] == "Test"
    assert row["asr_ms"] == "10.0"
    assert row["total_ms"] == "60.0"
    assert row["llm_tokens_in"] == "3"
    assert row["cost_est_usd"] == "0.001"
    assert row["error"] == ""
    assert int(row["ts"]) > 0


def test_feedback_evaluate_heuristics() -> None:
    state = ConversationState(session_id="s", persona_name="p")
    state.add_turn("user", "hello there")
    state.add_turn("assistant", "I understand, I am sorry about that.")
    state.add_turn("assistant", "Let us block your card and reissue a new one.")

    report = evaluate(state)
    assert "Greeting present" in report
    assert "Empathy detected" in report
    assert "Resolution provided" in report


def test_latency_logger_appends_without_rewriting_the_header(tmp_path) -> None:
    path = tmp_path / "latency.csv"
    logger = LatencyLogger(str(path))
    for turn in (1, 2, 3):
        logger.log_turn(
            turn=turn,
            persona="P",
            asr_ms=1.0,
            llm_ms=1.0,
            tts_ms=1.0,
            total_ms=3.0,
            input_chars=1,
            output_chars=1,
            llm_tokens_in=None,
            llm_tokens_out=None,
            asr_secs=0.1,
            tts_chars=1,
            cost_est_usd=None,
            error=None,
        )

    rows = list(csv.DictReader(path.read_text().splitlines()))
    assert [r["turn"] for r in rows] == ["1", "2", "3"]
    # Absent values are written as blanks, not the string "None".
    assert rows[0]["llm_tokens_in"] == ""
    assert rows[0]["cost_est_usd"] == ""


def test_latency_logger_reuses_an_existing_file(tmp_path) -> None:
    path = tmp_path / "latency.csv"
    LatencyLogger(str(path)).log_turn(
        turn=1,
        persona="P",
        asr_ms=1.0,
        llm_ms=1.0,
        tts_ms=1.0,
        total_ms=3.0,
        input_chars=1,
        output_chars=1,
        llm_tokens_in=1,
        llm_tokens_out=1,
        asr_secs=0.1,
        tts_chars=1,
        cost_est_usd=0.0,
        error=None,
    )
    # A second logger on the same path must not truncate the first one's rows.
    LatencyLogger(str(path))
    assert len(path.read_text().strip().splitlines()) == 2


def test_latency_logger_accepts_a_bare_filename(tmp_path, monkeypatch) -> None:
    # os.path.dirname("latency.csv") is "", and os.makedirs("") raises
    # FileNotFoundError rather than doing nothing.
    monkeypatch.chdir(tmp_path)
    LatencyLogger("latency.csv")
    assert (tmp_path / "latency.csv").exists()


def test_latency_logger_creates_missing_parent_directories(tmp_path) -> None:
    path = tmp_path / "nested" / "deeper" / "latency.csv"
    LatencyLogger(str(path))
    assert path.exists()


def test_latency_logger_records_an_error_string(tmp_path) -> None:
    path = tmp_path / "latency.csv"
    LatencyLogger(str(path)).log_turn(
        turn=1,
        persona="P",
        asr_ms=0.0,
        llm_ms=0.0,
        tts_ms=0.0,
        total_ms=0.0,
        input_chars=0,
        output_chars=0,
        llm_tokens_in=None,
        llm_tokens_out=None,
        asr_secs=0.0,
        tts_chars=0,
        cost_est_usd=None,
        error="ASR timed out",
    )
    first_row = next(iter(csv.DictReader(path.read_text().splitlines())))
    assert first_row["error"] == "ASR timed out"


def test_feedback_reports_what_is_missing() -> None:
    state = ConversationState(session_id="s", persona_name="p")
    state.add_turn("user", "my card is gone")
    state.add_turn("assistant", "Okay.")

    report = evaluate(state)
    assert "Missing greeting" in report
    assert "Verification missing" in report
    assert "Empathy missing" in report
    assert "Resolution unclear" in report


@pytest.mark.parametrize(
    "utterance",
    ["this is my problem", "which branch?", "shipping was delayed", "hit a snag"],
)
def test_a_word_containing_hi_is_not_a_greeting(utterance) -> None:
    # The pattern read as `\bhello` OR `hi` OR `good\s+(...)\b`: alternation
    # binds looser than the anchors, so the bare `hi` matched inside ordinary
    # words and almost every call scored a greeting it never got.
    state = ConversationState(session_id="s", persona_name="p")
    state.add_turn("user", utterance)
    assert "Missing greeting" in evaluate(state)


@pytest.mark.parametrize(
    "utterance",
    ["Hello there", "hi, I need help", "Good morning", "good evening to you", "Hey"],
)
def test_real_greetings_are_recognised(utterance) -> None:
    state = ConversationState(session_id="s", persona_name="p")
    state.add_turn("user", utterance)
    assert "Greeting present" in evaluate(state)


def test_only_the_opening_turns_count_as_a_greeting() -> None:
    # A greeting five turns in is not how the call opened.
    state = ConversationState(session_id="s", persona_name="p")
    for _ in range(3):
        state.add_turn("user", "and another thing")
    state.add_turn("user", "hello")

    assert "Missing greeting" in evaluate(state)


def test_feedback_on_an_empty_conversation_reports_everything_missing() -> None:
    report = evaluate(ConversationState(session_id="s", persona_name="p"))
    assert report.startswith("Post-run evaluation:")
    for missing in ("Missing greeting", "Verification missing", "Empathy missing"):
        assert missing in report
