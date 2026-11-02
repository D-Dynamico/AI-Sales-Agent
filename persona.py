"""Loads the sender persona from persona.toml.

TOML via stdlib `tomllib` (Python 3.11+) rather than YAML: no new dependency, and unlike
JSON it takes comments and multi-line strings — both of which a value proposition needs.

Environment variables win over the file so a deployment can differ without editing it.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PERSONA_FILE = Path(os.getenv("PERSONA_FILE", "persona.toml"))

# Mirrors persona.toml, so the pipeline still runs if the file is missing or unreadable.
_DEFAULTS = {
    "sender": {"name": "Rohan Verma", "company": "Kriya AI", "from_email": "rohan@kriya.ai"},
    "offering": {"description": "an AI infrastructure company that helps B2B teams "
                                "automate sales, marketing, and ops workflows"},
    "icp": {
        "description": "B2B teams with a revenue, growth or operations function",
        "functions": ["sales", "marketing", "revenue", "growth", "demand", "outbound",
                      "gtm", "operations", "ops", "business", "bizdev", "partnerships",
                      "commercial", "customer", "success", "strategy"],
        "disqualifying_titles": ["intern", "trainee", "student", "apprentice", "assistant",
                                 "associate", "recruiter", "volunteer", "retired",
                                 "freelance"],
    },
}


@dataclass(frozen=True)
class Persona:
    name: str
    company: str
    from_email: str
    offering: str
    icp_description: str
    icp_functions: frozenset = field(default_factory=frozenset)
    disqualifying_titles: frozenset = field(default_factory=frozenset)

    @property
    def signoff(self) -> str:
        return f"{self.name}, {self.company}"

    @property
    def from_name(self) -> str:
        return f"{self.name} from {self.company}"


def _read_file(path: Path) -> dict:
    """A malformed persona file must not take the pipeline down silently, but it must not
    pass unnoticed either — it changes who the email claims to be from."""
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ValueError(f"{path} could not be parsed: {exc}") from exc


def load(path: Path | None = None) -> Persona:
    data = _read_file(path or PERSONA_FILE)

    def section(name: str) -> dict:
        return {**_DEFAULTS[name], **(data.get(name) or {})}

    sender, offering, icp = section("sender"), section("offering"), section("icp")

    return Persona(
        name=os.getenv("SENDER_NAME") or sender["name"],
        company=os.getenv("SENDER_COMPANY") or sender["company"],
        from_email=os.getenv("FROM_EMAIL") or sender["from_email"],
        offering=os.getenv("SENDER_OFFERING") or offering["description"],
        icp_description=icp["description"],
        icp_functions=frozenset(icp["functions"]),
        disqualifying_titles=frozenset(icp["disqualifying_titles"]),
    )


PERSONA = load()
