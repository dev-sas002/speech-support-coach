import argparse
import os
import uuid

from dotenv import load_dotenv

from src import feedback as feedback_module
from src.logger import LatencyLogger, init_logger
from src.persona_loader import load_persona
from src.voice_client import VoiceClient


def main() -> None:
    load_dotenv()
    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persona",
        choices=["card_lost", "transfer_failed", "account_locked"],
        default="card_lost",
    )
    parser.add_argument("--turns", type=int, default=3)
    args = parser.parse_args()

    persona = load_persona(args.persona)

    session_id = str(uuid.uuid4())
    vc = VoiceClient(persona=persona, session_id=session_id)
    logger = LatencyLogger(path=os.path.join("logs", "latency_log.csv"))
    vc.run(max_turns=args.turns, logger_obj=logger, feedback=feedback_module)


if __name__ == "__main__":
    main()
