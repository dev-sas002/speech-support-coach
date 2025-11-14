"""
Persona loading, and the link between the shipped personas and the offline
script they select.
"""

from __future__ import annotations

import json

import pytest

from src.local_backend import SCRIPTS, detect_scenario
from src.persona_loader import (
    PERSONA_DISPLAY_NAMES,
    PERSONA_FILES,
    get_personas_dir,
    list_personas,
    load_persona,
)

REQUIRED_FIELDS = ("name", "scenario", "system_prompt")


class TestShippedPersonas:
    @pytest.mark.parametrize("key", list(PERSONA_FILES))
    def test_every_declared_persona_exists_and_is_complete(self, key, project_root):
        persona = load_persona(key, project_root)
        for field in REQUIRED_FIELDS:
            assert persona[field], f"{key} has an empty {field}"

    def test_every_declared_persona_has_a_display_name(self):
        assert set(PERSONA_FILES) == set(PERSONA_DISPLAY_NAMES)

    def test_listing_returns_all_of_them(self, project_root):
        assert {key for key, _ in list_personas(project_root)} == set(PERSONA_FILES)

    def test_the_cli_choices_match_the_persona_keys(self, project_root):
        # main.py restricts --persona to these three; a drift here means the
        # CLI silently cannot reach a persona that exists.
        source = (project_root / "main.py").read_text()
        for key in PERSONA_FILES:
            assert f'"{key}"' in source

    @pytest.mark.parametrize("key", list(PERSONA_FILES))
    def test_each_shipped_persona_selects_its_own_offline_script(self, key, project_root):
        # The offline agent picks a script by matching words in the system
        # prompt. If a prompt is reworded past those hints, the trainee gets
        # another scenario's procedure with no indication anything is wrong.
        persona = load_persona(key, project_root)
        messages = [{"role": "system", "content": persona["system_prompt"]}]
        assert detect_scenario(messages) == key

    def test_every_scenario_has_a_script(self):
        assert set(SCRIPTS) == set(PERSONA_FILES)


class TestLoadPersona:
    def test_a_missing_persona_names_the_path_it_looked_for(self, tmp_path):
        (tmp_path / "config" / "personas").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="nope"):
            load_persona("nope", tmp_path)

    @pytest.mark.parametrize("missing", REQUIRED_FIELDS)
    def test_an_incomplete_persona_is_rejected_by_name(self, tmp_path, missing):
        personas = tmp_path / "config" / "personas"
        personas.mkdir(parents=True)
        data = dict.fromkeys(REQUIRED_FIELDS, "x")
        del data[missing]
        (personas / "card_lost.json").write_text(json.dumps(data))

        with pytest.raises(ValueError, match=missing):
            load_persona("card_lost", tmp_path)

    def test_an_unknown_key_falls_back_to_a_matching_filename(self, tmp_path):
        personas = tmp_path / "config" / "personas"
        personas.mkdir(parents=True)
        data = dict.fromkeys(REQUIRED_FIELDS, "x")
        (personas / "custom.json").write_text(json.dumps(data))

        assert load_persona("custom", tmp_path) == data

    def test_malformed_json_is_not_swallowed(self, tmp_path):
        personas = tmp_path / "config" / "personas"
        personas.mkdir(parents=True)
        (personas / "card_lost.json").write_text("{ not json")

        with pytest.raises(json.JSONDecodeError):
            load_persona("card_lost", tmp_path)


class TestListPersonas:
    def test_personas_that_are_not_on_disk_are_skipped(self, tmp_path):
        personas = tmp_path / "config" / "personas"
        personas.mkdir(parents=True)
        (personas / "card_lost.json").write_text(json.dumps(dict.fromkeys(REQUIRED_FIELDS, "x")))

        assert [key for key, _ in list_personas(tmp_path)] == ["card_lost"]

    def test_an_empty_directory_lists_nothing_rather_than_failing(self, tmp_path):
        (tmp_path / "config" / "personas").mkdir(parents=True)
        assert list_personas(tmp_path) == []


class TestPersonasDir:
    def test_it_defaults_to_the_project_root(self, project_root, personas_dir):
        assert get_personas_dir() == personas_dir
        assert get_personas_dir(project_root) == personas_dir

    def test_it_is_derived_from_a_given_base(self, tmp_path):
        assert get_personas_dir(tmp_path) == tmp_path / "config" / "personas"
