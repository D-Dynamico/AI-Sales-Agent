"""Tests for persona configuration.

The persona decides who the email claims to be from, so the failure that matters most is a
silent one: a typo'd or unreadable config that quietly falls back to someone else's
identity. These tests pin that it either loads what you wrote or complains loudly.
"""

import pytest

from persona import Persona, load

SAMPLE = """
[sender]
name = "Sam"
company = "Northwind Labs"
from_email = "sam@northwind.example"

[offering]
description = "a platform that reconciles invoices automatically"

[icp]
description = "finance teams"
functions = ["finance", "accounting"]
disqualifying_titles = ["intern"]
"""


def _write(tmp_path, text):
    path = tmp_path / "persona.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_every_field(tmp_path):
    persona = load(_write(tmp_path, SAMPLE))

    assert persona.name == "Sam"
    assert persona.company == "Northwind Labs"
    assert persona.from_email == "sam@northwind.example"
    assert "reconciles invoices" in persona.offering
    assert persona.icp_functions == frozenset({"finance", "accounting"})


def test_derived_identity_strings():
    persona = Persona("Sam", "Northwind Labs", "s@n.example", "offering", "icp")
    assert persona.signoff == "Sam, Northwind Labs"
    assert persona.from_name == "Sam from Northwind Labs"


class TestOverridesAndFallbacks:
    def test_environment_beats_the_file(self, tmp_path, monkeypatch):
        """Deployments differ without editing a tracked file."""
        monkeypatch.setenv("SENDER_COMPANY", "Env Corp")
        monkeypatch.setenv("FROM_EMAIL", "env@example.com")

        persona = load(_write(tmp_path, SAMPLE))

        assert persona.company == "Env Corp"
        assert persona.from_email == "env@example.com"
        assert persona.name == "Sam"  # untouched keys still come from the file

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        persona = load(tmp_path / "absent.toml")
        assert persona.company
        assert persona.icp_functions

    def test_partial_file_fills_the_gaps(self, tmp_path):
        """Setting one field must not blank out every field you didn't mention."""
        persona = load(_write(tmp_path, '[sender]\nname = "Solo"\n'))

        assert persona.name == "Solo"
        assert persona.company
        assert persona.from_email
        assert persona.icp_functions

    def test_malformed_file_raises_rather_than_impersonating(self, tmp_path):
        """A broken config silently reverting to defaults would send mail signed by the
        wrong company — louder is safer than convenient here."""
        with pytest.raises(ValueError, match="could not be parsed"):
            load(_write(tmp_path, "[sender\nname = broken"))


class TestPersonaReachesTheProject:
    def test_drafter_prompt_uses_the_persona(self):
        from agents.drafter import SYSTEM_PROMPT
        from persona import PERSONA

        assert PERSONA.company in SYSTEM_PROMPT
        assert PERSONA.signoff in SYSTEM_PROMPT
        assert PERSONA.offering in SYSTEM_PROMPT

    def test_sender_headers_use_the_persona(self):
        from persona import PERSONA
        from tools import sender

        assert sender.FROM_NAME == PERSONA.from_name
        assert sender.FROM_EMAIL == PERSONA.from_email

    def test_icp_lists_come_from_the_persona(self):
        from agents.icp import DISQUALIFYING_TITLES, RELEVANT_FUNCTIONS
        from persona import PERSONA

        assert RELEVANT_FUNCTIONS == set(PERSONA.icp_functions)
        assert DISQUALIFYING_TITLES == set(PERSONA.disqualifying_titles)

    def test_no_hardcoded_identity_left_in_source(self):
        """The point of the change: swapping identity is a config edit, not a prompt edit."""
        import pathlib

        for name in ["agents/drafter.py", "tools/sender.py", "app.py", "agents/icp.py"]:
            source = pathlib.Path(name).read_text(encoding="utf-8")
            assert "Alchemyst" not in source, f"{name} still hardcodes the company name"
