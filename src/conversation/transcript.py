"""
Exporting a call.

A support-training call is only useful if it can leave the browser: a reviewer
wants the transcript next to the latency it was produced at, and a Streamlit
session state is gone the moment the tab closes. JSON is for tooling, Markdown
is for a human reading it in a pull request.

Both formats carry the metrics, because a transcript without them loses the
half of this project that is interesting.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .engine import ConversationEngine


def _header(engine: ConversationEngine) -> dict[str, Any]:
    return {
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "persona": engine.persona.get("name", "Assistant"),
        "scenario": engine.persona.get("scenario", ""),
        "provider": getattr(engine.llm, "name", "unknown"),
        "budget_ms": engine.budget_ms,
    }


def export_json(engine: ConversationEngine, review: str | None = None) -> str:
    """The whole call as JSON: header, turns with metrics, latency summary."""
    payload: dict[str, Any] = {
        **_header(engine),
        "summary": engine.summary(),
        "turns": [
            {
                "index": turn.metrics.index,
                "customer": turn.user_text,
                "agent": turn.assistant_text,
                "metrics": turn.metrics.as_dict(),
            }
            for turn in engine.turns
        ],
    }
    if review:
        payload["review"] = review
    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_markdown(engine: ConversationEngine, review: str | None = None) -> str:
    """The whole call as Markdown, readable without any tooling."""
    header = _header(engine)
    summary = engine.summary()
    lines = [
        f"# Support call — {header['persona']}",
        "",
        f"- **Scenario:** {header['scenario'] or 'unspecified'}",
        f"- **Model provider:** `{header['provider']}`",
        f"- **Exported:** {header['exported_at']}",
    ]
    if summary.get("turns"):
        lines += [
            f"- **Turns:** {summary['turns']}",
            f"- **Mean time to first response:** {summary['mean_response_ms']:.0f} ms"
            f" (budget {summary['budget_ms']:.0f} ms,"
            f" {summary['over_budget']} over)",
        ]
    lines += ["", "## Transcript", ""]

    for turn in engine.turns:
        metrics = turn.metrics
        lines += [
            f"### Turn {metrics.index}",
            "",
            f"**Customer:** {turn.user_text}",
            "",
            f"**{header['persona']}:** {turn.assistant_text}",
            "",
            f"_Responded in {metrics.responsiveness_ms:.0f} ms"
            f"{' — over budget' if metrics.over_budget else ''}._",
            "",
        ]

    if review:
        lines += ["## Call review", "", review, ""]
    return "\n".join(lines)
