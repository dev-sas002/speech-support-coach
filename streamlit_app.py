"""
The Streamlit front end.

This module is a view and nothing else. It decides what to draw and when to
rerun; every decision about the conversation — which provider answers, what the
prompt window holds, how a turn is timed, how a call is reviewed — belongs to
`src.conversation`, `src.providers` and `src.analysis`, and is reached through
them.

That separation is the reason the turn loop can be benchmarked from a script
and tested without a browser. It is also why this file can be read in one
sitting: there is no business logic hiding in a render function.

Two modes share almost all of it. Voice mode drives a microphone through
`SimpleVoiceHandler`; text mode types into the same conversation engine. Text
mode is what a container, a CI runner or a cloud host gets, because none of
them has audio hardware — and it exercises everything except ASR and TTS.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.analysis import review_call
from src.conversation import (
    ConversationEngine,
    export_json,
    export_markdown,
    seed_demo_call,
    turn_budget_ms,
)
from src.persona_loader import PERSONA_DISPLAY_NAMES, list_personas
from src.providers import ProviderSet, create_provider_set
from src.state_manager import ConversationState

APP_TITLE = "Voice Support Simulator"
BASE_DIR = Path(__file__).parent

STYLES = """
<style>
:root {
  --vs-ink: #101828;
  --vs-muted: #667085;
  --vs-line: #e4e7ec;
  --vs-surface: #ffffff;
  --vs-accent: #4338ca;
  --vs-good: #067647;
  --vs-good-bg: #ecfdf3;
  --vs-warn: #b54708;
  --vs-warn-bg: #fffaeb;
  --vs-idle-bg: #f2f4f7;
}
.vs-hero {
  border: 1px solid var(--vs-line);
  border-left: 4px solid var(--vs-accent);
  border-radius: 12px;
  padding: 18px 22px;
  margin-bottom: 18px;
  background: var(--vs-surface);
}
.vs-hero h1 {
  font-size: 1.5rem; margin: 0 0 4px 0; color: var(--vs-ink); letter-spacing: -0.01em;
}
.vs-hero p { margin: 0; color: var(--vs-muted); font-size: 0.93rem; line-height: 1.5; }
.vs-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.vs-chip {
  font-size: 0.74rem; font-weight: 600; letter-spacing: 0.02em;
  padding: 4px 10px; border-radius: 999px; border: 1px solid var(--vs-line);
  background: var(--vs-idle-bg); color: var(--vs-muted);
}
.vs-chip.ok { background: var(--vs-good-bg); color: var(--vs-good); border-color: #abefc6; }
.vs-chip.warn { background: var(--vs-warn-bg); color: var(--vs-warn); border-color: #fedf89; }
.vs-provider {
  border: 1px solid var(--vs-line); border-radius: 10px;
  padding: 10px 12px; margin-bottom: 8px; background: var(--vs-surface);
}
.vs-provider .kind {
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--vs-muted);
}
.vs-provider .name { font-size: 0.95rem; font-weight: 650; color: var(--vs-ink); }
.vs-provider .detail { font-size: 0.78rem; color: var(--vs-muted); line-height: 1.4; }
.vs-turnline {
  font-size: 0.76rem; color: var(--vs-muted); margin: -6px 0 14px 52px;
  font-variant-numeric: tabular-nums;
}
.vs-turnline b { color: var(--vs-ink); font-weight: 650; }
.vs-turnline .over { color: var(--vs-warn); font-weight: 650; }
.vs-empty {
  border: 1px dashed var(--vs-line); border-radius: 12px; padding: 26px;
  text-align: center; color: var(--vs-muted); font-size: 0.9rem; background: var(--vs-surface);
}
div[data-testid="stSidebar"] h2 { font-size: 0.95rem; }
/* Streamlit's default top padding leaves the page opening on empty space. */
div[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; }
</style>
"""


# --- persona and session wiring -------------------------------------------


def load_personas() -> list[tuple[str, dict[str, Any]]]:
    """Available personas as (display name, persona) pairs, for the selector."""
    return [
        (PERSONA_DISPLAY_NAMES.get(key, data.get("name", key)), data)
        for key, data in list_personas(BASE_DIR)
    ]


def text_mode_enabled() -> bool:
    """Text mode is forced by VOICE_IO=text, or selected when there is no mic."""
    setting = (os.getenv("VOICE_IO") or "auto").lower()
    if setting == "text":
        return True
    if setting == "voice":
        return False

    from src.audio_devices import audio_available

    return not audio_available()


def get_providers() -> ProviderSet:
    """
    Resolve the three providers once per session.

    Building them per rerun would construct a new HTTP client — and a new
    connection pool — on every keystroke, and would re-probe the audio stack
    each time the page redrew.
    """
    if "providers" not in st.session_state:
        st.session_state.providers = create_provider_set()
    return st.session_state.providers


def get_engine(persona: dict[str, Any], display_name: str) -> ConversationEngine:
    """The engine for the selected persona, rebuilt when the scenario changes."""
    if st.session_state.get("engine_persona") != display_name:
        st.session_state.engine_persona = display_name
        st.session_state.engine = ConversationEngine(
            persona=persona,
            llm=get_providers().llm,
            state=ConversationState(
                session_id=f"ui-{display_name}",
                persona_name=persona.get("name", "Agent"),
            ),
        )
        if os.getenv("SEED_DEMO_CALL", "0") == "1":
            # The container sets this so a first boot lands on a real
            # conversation rather than an empty page. The replies are produced
            # by the engine at seed time, not stored: the transcript and its
            # latencies are as real as any other turn.
            seed_demo_call(st.session_state.engine)
        st.session_state.review = None
    return st.session_state.engine


# --- presentation pieces ---------------------------------------------------


def render_hero(providers: ProviderSet, mode: str) -> None:
    ready = {s.kind: s for s in providers.statuses()}
    chips = [
        f'<span class="vs-chip {"ok" if ready[kind].ready else "warn"}">'
        f"{label} · {ready[kind].resolved}</span>"
        for kind, label in (("stt", "STT"), ("llm", "LLM"), ("tts", "TTS"))
    ]
    chips.append(f'<span class="vs-chip">{mode}</span>')
    chips.append(f'<span class="vs-chip">budget {turn_budget_ms():.0f} ms</span>')

    st.markdown(
        f"""
        <div class="vs-hero">
          <h1>{APP_TITLE}</h1>
          <p>A bank support call, end to end: transcribe, decide, speak — with
          every turn timed. You are the customer; the assistant plays the
          support agent for the selected scenario.</p>
          <div class="vs-chips">{"".join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_provider_panel(providers: ProviderSet) -> None:
    st.markdown("## Pipeline")
    labels = {"stt": "Speech to text", "llm": "Language model", "tts": "Text to speech"}
    for status in providers.statuses():
        badge = "✅" if status.ready else "⚠️"
        st.markdown(
            f"""
            <div class="vs-provider">
              <div class="kind">{labels[status.kind]}</div>
              <div class="name">{badge} {status.resolved}</div>
              <div class="detail">{status.detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    for warning in providers.warnings:
        st.warning(warning, icon="⚠️")


def ms(value: float) -> str:
    """
    Format a duration.

    The scripted agent answers in well under a millisecond. Rounding that to
    "0 ms" reads like a broken metric rather than a fast one.
    """
    if value >= 10:
        return f"{value:.0f} ms"
    if value >= 1:
        return f"{value:.1f} ms"
    return f"{value:.2f} ms"


def render_turn_line(metrics) -> None:
    parts = [f"turn {metrics.index}"]
    if metrics.first_token_ms is not None:
        parts.append(f"first token <b>{ms(metrics.first_token_ms)}</b>")
    if metrics.asr_ms:
        parts.append(f"ASR {ms(metrics.asr_ms)}")
    parts.append(f"reply {ms(metrics.llm_ms)}")
    if metrics.tokens_out:
        parts.append(f"{metrics.tokens_out} tokens out")
    verdict = '<span class="over">over budget</span>' if metrics.over_budget else "within budget"
    st.markdown(
        f'<div class="vs-turnline">{" · ".join(parts)} · {verdict}</div>',
        unsafe_allow_html=True,
    )


def render_transcript(engine: ConversationEngine) -> None:
    if not engine.turns:
        st.markdown(
            '<div class="vs-empty">No turns yet — say what the customer '
            "wants in the box below and the agent will answer.</div>",
            unsafe_allow_html=True,
        )
        return

    for turn in engine.turns:
        with st.chat_message("user"):
            st.write(turn.user_text)
        with st.chat_message("assistant"):
            st.write(turn.assistant_text)
        render_turn_line(turn.metrics)


def render_latency_panel(engine: ConversationEngine) -> None:
    summary = engine.summary()
    if not summary.get("turns"):
        return

    st.markdown("### Turn latency")
    columns = st.columns(4)
    columns[0].metric("Turns", summary["turns"])
    columns[1].metric("Mean first response", ms(summary["mean_response_ms"]))
    columns[2].metric("Worst first response", ms(summary["worst_response_ms"]))
    columns[3].metric(
        f"Over {summary['budget_ms']:.0f} ms budget",
        f"{summary['over_budget']} of {summary['turns']}",
    )

    with st.expander("Per-turn breakdown", expanded=False):
        st.dataframe(
            [
                {
                    "turn": row["index"],
                    "first token": ms(row["first_token_ms"])
                    if row["first_token_ms"] is not None
                    else "n/a",
                    "whole reply": ms(row["llm_ms"]),
                    "tokens in": row["tokens_in"],
                    "tokens out": row["tokens_out"],
                    # A boolean column renders as a checkbox, which reads as an
                    # interactive control rather than a result.
                    "budget": "OVER" if row["over_budget"] else "within",
                }
                for row in engine.metrics_table()
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Responsiveness is measured to the first token, not to the last. A "
            "long answer that starts immediately feels fast; a short one that "
            "starts late does not."
        )


def render_review(engine: ConversationEngine, providers: ProviderSet) -> None:
    st.markdown("### Call review")
    disabled = not engine.turns
    if st.button("Review this call", disabled=disabled, width="stretch"):
        st.session_state.review = review_call(
            engine.state,
            llm=providers.llm,
            persona_name=engine.persona.get("name", "Agent"),
        )

    review = st.session_state.get("review")
    if review is None:
        st.caption(
            "Scores the agent's side of the call and scans it for compliance "
            "breaches. The compliance scan is a fixed rule set and always "
            "runs; the qualitative part uses the language model when one is "
            "reachable and a keyword heuristic when it is not."
        )
        return

    if review.source == "heuristic":
        st.info(
            "No language model is reachable, so this is the offline keyword "
            "heuristic — a pattern match, not an assessment.",
            icon="ℹ️",  # noqa: RUF001
        )
    st.markdown(review.to_markdown())


def render_export(engine: ConversationEngine) -> None:
    st.markdown("## Export")
    if not engine.turns:
        st.caption("Take a turn first — there is nothing to export yet.")
        return

    review = st.session_state.get("review")
    review_text = review.to_markdown() if review else None
    scenario = engine.persona.get("scenario", "call").replace(" ", "_").lower()

    st.download_button(
        "Transcript (Markdown)",
        data=export_markdown(engine, review_text),
        file_name=f"{scenario}_call.md",
        mime="text/markdown",
        width="stretch",
    )
    st.download_button(
        "Transcript + metrics (JSON)",
        data=export_json(engine, review_text),
        file_name=f"{scenario}_call.json",
        mime="application/json",
        width="stretch",
    )


# --- the two modes ---------------------------------------------------------


def render_text_mode(personas: list[tuple[str, dict[str, Any]]]) -> None:
    """A typed conversation through the same engine voice mode uses."""
    providers = get_providers()

    with st.sidebar:
        st.markdown("## Scenario")
        names = [name for name, _ in personas]
        selected_name = st.selectbox("Support scenario", names, key="text_persona")
        persona = dict(personas)[selected_name]
        st.caption(persona.get("scenario", ""))
        render_provider_panel(providers)

    engine = get_engine(persona, selected_name)

    with st.sidebar:
        render_export(engine)
        if st.button("New call", width="stretch"):
            engine.reset()
            st.session_state.review = None
            st.rerun()

    render_hero(providers, "text mode")
    render_transcript(engine)

    prompt = st.chat_input("Type what the customer says…")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            try:
                st.write_stream(engine.stream_respond(prompt))
            except Exception as exc:  # surfaced, not swallowed
                st.error(f"The assistant could not respond: {exc}")
                return
        st.session_state.review = None
        st.rerun()

    if engine.turns:
        st.markdown("---")
        render_latency_panel(engine)
        st.markdown("---")
        render_review(engine, providers)


def render_voice_mode(personas: list[tuple[str, dict[str, Any]]]) -> None:
    """Push-to-talk against a real microphone, on a machine that has one."""
    from src.simple_voice_handler import SimpleVoiceHandler

    providers = get_providers()

    with st.sidebar:
        st.markdown("## Scenario")
        names = [name for name, _ in personas]
        selected_name = st.selectbox("Support scenario", names, key="voice_persona")
        persona = dict(personas)[selected_name]
        st.caption(persona.get("scenario", ""))
        render_provider_panel(providers)

    if st.session_state.get("voice_persona_active") != selected_name:
        existing = st.session_state.get("voice_handler")
        if existing is not None:
            existing.cleanup()
        st.session_state.voice_persona_active = selected_name
        st.session_state.voice_handler = SimpleVoiceHandler(persona)
        st.session_state.voice_metrics = []
        st.session_state.voice_status = "Ready"

    handler = st.session_state.voice_handler

    render_hero(providers, "voice mode")
    st.info(st.session_state.voice_status, icon="🎙️")

    left, right = st.columns(2)
    if handler.is_recording:
        if right.button("Stop and send", type="primary", width="stretch"):
            with st.spinner("Transcribing, answering, speaking…"):
                metrics = handler.process_voice_input()
            if "error" in metrics:
                st.session_state.voice_status = f"Error: {metrics['error']}"
            else:
                st.session_state.voice_metrics.append(metrics)
                st.session_state.voice_status = "Ready"
            st.rerun()
    elif left.button("Start recording", type="primary", width="stretch"):
        handler.start_recording()
        st.session_state.voice_status = "Recording — speak now"
        st.rerun()

    if right.button("New call", width="stretch"):
        handler.reset_conversation()
        st.session_state.voice_metrics = []
        st.rerun()

    st.markdown("---")
    if handler.state.turns:
        for turn in handler.state.turns:
            with st.chat_message("user" if turn["role"] == "user" else "assistant"):
                st.write(turn["text"])
    else:
        st.markdown(
            '<div class="vs-empty">Press <b>Start recording</b>, say what the '
            "customer wants, then press <b>Stop and send</b>.</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.voice_metrics:
        st.markdown("---")
        st.markdown("### Turn latency")
        latest = st.session_state.voice_metrics[-1]
        columns = st.columns(4)
        columns[0].metric("ASR", ms(latest.get("asr_ms", 0)))
        first_audio = latest.get("first_audio_ms")
        columns[1].metric("First audio", "n/a" if first_audio is None else ms(first_audio))
        columns[2].metric("Reply", ms(latest.get("llm_ms", 0)))
        columns[3].metric("Turn", ms(latest.get("total_ms", 0)))
        st.caption(
            "Synthesis starts at the first sentence boundary, so the reply and "
            "speech timings overlap and do not sum to the turn."
        )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, page_icon="🎙️", layout="wide")
    st.markdown(STYLES, unsafe_allow_html=True)

    personas = load_personas()
    if not personas:
        st.error("No personas found — check config/personas/.")
        return

    if text_mode_enabled():
        render_text_mode(personas)
    else:
        render_voice_mode(personas)


if __name__ == "__main__":
    main()
