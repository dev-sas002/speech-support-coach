import csv
import os
import time

from loguru import logger


def init_logger():
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))


class LatencyLogger:
    def __init__(self, path: str):
        self.path = path
        # A bare filename ("latency.csv") has an empty dirname, and
        # os.makedirs("") raises FileNotFoundError rather than doing nothing.
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "ts",
                        "turn",
                        "persona",
                        "asr_ms",
                        "llm_ms",
                        "tts_ms",
                        "total_ms",
                        "input_chars",
                        "output_chars",
                        "llm_tokens_in",
                        "llm_tokens_out",
                        "asr_secs",
                        "tts_chars",
                        "cost_est_usd",
                        "error",
                    ]
                )

    def log_turn(
        self,
        turn: int,
        persona: str,
        asr_ms: float,
        llm_ms: float,
        tts_ms: float,
        total_ms: float,
        input_chars: int,
        output_chars: int,
        llm_tokens_in: int | None,
        llm_tokens_out: int | None,
        asr_secs: float,
        tts_chars: int,
        cost_est_usd: float | None,
        error: str | None,
    ):
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    int(time.time() * 1000),
                    turn,
                    persona,
                    round(asr_ms, 1),
                    round(llm_ms, 1),
                    round(tts_ms, 1),
                    round(total_ms, 1),
                    input_chars,
                    output_chars,
                    llm_tokens_in or "",
                    llm_tokens_out or "",
                    round(asr_secs, 2),
                    tts_chars,
                    (None if cost_est_usd is None else round(cost_est_usd, 6)),
                    error or "",
                ]
            )
