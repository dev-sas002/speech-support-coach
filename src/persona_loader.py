from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Canonical persona keys used across the project (CLI args, UI, tests).
PERSONA_FILES: dict[str, str] = {
    "card_lost": "card_lost.json",
    "transfer_failed": "transfer_failed.json",
    "account_locked": "account_locked.json",
}

PERSONA_DISPLAY_NAMES: dict[str, str] = {
    "card_lost": "Lost Card",
    "transfer_failed": "Failed Transfer",
    "account_locked": "Locked Account",
}


def _default_base_dir() -> Path:
    """Return the project root directory."""
    # This file lives in src/, so the project root is one level up.
    return Path(__file__).resolve().parent.parent


def get_personas_dir(base_dir: Path | None = None) -> Path:
    """
    Compute the personas directory from a given base directory.

    If no base directory is provided, the project root is used.
    """
    base = base_dir or _default_base_dir()
    return base / "config" / "personas"


def load_persona(key: str, base_dir: Path | None = None) -> dict[str, Any]:
    """
    Load and minimally validate a persona by its canonical key.

    Raises FileNotFoundError if the persona file is missing and ValueError
    if required fields are not present.
    """
    personas_dir = get_personas_dir(base_dir)
    filename = PERSONA_FILES.get(key, f"{key}.json")
    path = personas_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Persona file not found for key '{key}': {path}")

    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    for field_name in ("name", "scenario", "system_prompt"):
        if field_name not in data:
            raise ValueError(f"Persona '{key}' missing required field '{field_name}'")

    return data


def list_personas(base_dir: Path | None = None) -> list[tuple[str, dict[str, Any]]]:
    """
    Return a list of (key, persona_dict) for all known personas that exist on disk.
    """
    personas_dir = get_personas_dir(base_dir)
    items: list[tuple[str, dict[str, Any]]] = []

    for key, filename in PERSONA_FILES.items():
        path = personas_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        items.append((key, data))

    return items
