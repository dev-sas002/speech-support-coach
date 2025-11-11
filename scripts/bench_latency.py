#!/usr/bin/env python3
"""
Measure the turn loop.

Runs a fixed script of customer lines through `ConversationEngine` against a
chosen provider and reports the distribution of *time to first token* — when
the caller first hears something — alongside whole-turn latency.

The point is to make the claim in the README checkable. Run it against the
scripted provider and you are measuring this project's own overhead: prompt
assembly, the window, the streaming machinery. Run it with a key and
`--provider groq` and you are measuring the provider. Both numbers are useful;
conflating them is not.

    python scripts/bench_latency.py --turns 12
    python scripts/bench_latency.py --provider groq --turns 12
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Run from anywhere: the project root is not on the path when this file is
# invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conversation import ConversationEngine
from src.conversation.seed import openings_for
from src.persona_loader import load_persona
from src.providers import create_llm


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None, help="LLM provider name (default: auto)")
    parser.add_argument(
        "--persona",
        default="card_lost",
        choices=["card_lost", "transfer_failed", "account_locked"],
    )
    parser.add_argument("--turns", type=int, default=12)
    args = parser.parse_args()

    persona = load_persona(args.persona)
    llm = create_llm(args.provider)
    engine = ConversationEngine(persona=persona, llm=llm)

    lines = openings_for(persona)
    first_token: list[float] = []
    whole_turn: list[float] = []

    started = time.perf_counter()
    for index in range(args.turns):
        # The engine's rolling window means a long run is not a growing prompt;
        # cycling the same lines keeps the input comparable turn to turn.
        text = lines[index % len(lines)]
        turn_started = time.perf_counter()
        for _ in engine.stream_respond(text):
            pass
        whole_turn.append((time.perf_counter() - turn_started) * 1000)
        metrics = engine.last_turn.metrics
        if metrics.first_token_ms is not None:
            first_token.append(metrics.first_token_ms)
    elapsed = time.perf_counter() - started

    print(f"provider          {getattr(llm, 'name', 'unknown')}")
    print(f"persona           {persona.get('name')}")
    print(f"turns             {args.turns} in {elapsed:.2f}s")
    print(f"prompt window     {engine.state.total_chars} chars held after the run")
    print()
    print("                    p50        p95       max")
    for label, series in (("time to first token", first_token), ("whole turn", whole_turn)):
        if not series:
            continue
        print(
            f"{label:<20}{statistics.median(series):>7.1f}ms"
            f"{percentile(series, 0.95):>9.1f}ms"
            f"{max(series):>9.1f}ms"
        )
    over = sum(1 for value in first_token if value > engine.budget_ms)
    print()
    print(f"budget            {engine.budget_ms:.0f}ms — {over}/{len(first_token)} turns over")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
