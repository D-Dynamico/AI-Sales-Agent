"""The draft → critique → rewrite loop.

Lives here rather than in main.py so the CLI, the Streamlit app and the eval harness all
exercise the identical gate. If this logic were duplicated per caller, the eval harness
would eventually be measuring something the CLI doesn't actually do.
"""

from dataclasses import dataclass

from agents.critic import score_draft
from agents.drafter import draft_email, rewrite_email
from agents.schemas import CriticVerdict
from config import MAX_DRAFT_ATTEMPTS


@dataclass
class GatedDraft:
    draft: str
    verdict: CriticVerdict
    attempts: int

    @property
    def approved(self) -> bool:
        return self.verdict.passed

    @property
    def status(self) -> str:
        return "APPROVED" if self.approved else "NEEDS_REVIEW"


def draft_with_gate(lead: dict, research: str, sources: list[str],
                    max_attempts: int = MAX_DRAFT_ATTEMPTS) -> GatedDraft:
    """Draft, check, and rewrite up to `max_attempts` times.

    Returns the best attempt along with its verdict. A draft that never passes is returned
    anyway — flagged, not discarded — because a human reviewing a rejected draft plus the
    reasons it was rejected is more useful than an empty result.
    """
    draft = draft_email(lead, research)
    verdict = score_draft(draft, sources, lead=lead)
    attempts = 1

    while not verdict.passed and attempts < max_attempts:
        draft = rewrite_email(lead, research, previous_draft=draft, critique=verdict.critique)
        verdict = score_draft(draft, sources, lead=lead)
        attempts += 1

    return GatedDraft(draft=draft, verdict=verdict, attempts=attempts)
