"""Structured types shared across the pipeline."""

from typing import Literal

from typing import get_args

from pydantic import BaseModel, Field, field_validator

TriggerKind = Literal[
    "funding", "hiring", "product_launch", "expansion", "leadership_change",
    "partnership", "financials", "other",
]


class TriggerEvent(BaseModel):
    """A specific, recent thing that happened, worth opening an email on."""

    kind: TriggerKind = "other"
    summary: str = Field(description="One sentence, specific, no hedging")
    source_quote: str = Field(
        default="",
        description="Verbatim text from the search results that reports this event",
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _unknown_kind_is_other(cls, value):
        """Models invent labels ("fundraising_advisors", "corporate_restructuring"). A strict
        Literal made one such label fail validation for the *whole* response, discarding every
        well-quoted event alongside it and leaving the brief empty. The label is the least
        important part of an event, so coerce it rather than lose the event."""
        if isinstance(value, str):
            normalised = value.strip().lower().replace(" ", "_").replace("-", "_")
            if normalised in get_args(TriggerKind):
                return normalised
        return "other"


class ResearchResult(BaseModel):
    """Structured output of the enrichment step.

    Replaces the free-prose brief as the *internal* representation. A brief is still
    rendered from this for the drafter, but structure means the pipeline can reason about
    what it found — how much, of what kind, and how confident — instead of pattern-matching
    on sentences.
    """

    company: str = ""
    trigger_events: list[TriggerEvent] = Field(default_factory=list)
    pain_hypotheses: list[str] = Field(
        default_factory=list,
        description="Plausible problems implied by the events — clearly inference, not fact",
    )
    confidence: float = Field(
        default=0.0,
        description="0-1, how much usable, recent, specific signal the research actually found",
    )
    raw_sources: list[str] = Field(default_factory=list)

    def to_brief(self) -> str:
        """Flatten to the prose the drafter consumes.

        Pain hypotheses are labelled as inference so the drafter doesn't restate them as
        facts about the company — the critic would reject those, correctly, since no source
        supports them.
        """
        if not self.trigger_events:
            if self.raw_sources:
                return (f"Found {len(self.raw_sources)} recent search results, but no specific, "
                        "quotable events could be extracted from them.")
            return "No recent search results were found for this company."

        lines = [f"- {e.summary}" for e in self.trigger_events]
        if self.pain_hypotheses:
            lines.append("Likely challenges (inference, not established fact): "
                         + "; ".join(self.pain_hypotheses))
        return "\n".join(lines)


class EmailDraft(BaseModel):
    """A drafted email as two fields rather than one blob.

    Replaces regex-parsing "Subject:" back out of free text in the sender. The subject was
    never really part of the body — reconstructing it downstream meant a model that forgot
    the header produced an email with no subject and no way to tell until send time.
    """

    subject: str = Field(default="", description="Punchy, specific, under 8 words")
    body: str = Field(default="", description="The email body only — never the subject line")

    def render(self) -> str:
        """The classic 'Subject: ...' form, for logs, previews and format checks."""
        return f"Subject: {self.subject}\n\n{self.body}".strip()


class FactualClaim(BaseModel):
    """One checkable assertion the draft makes about the recipient's company or person."""

    text: str = Field(description="The claim as the email states it")
    grounded: bool = Field(description="True only if a source snippet supports it")
    source_quote: str = Field(
        default="",
        description="Verbatim text from the search results that supports the claim. "
                    "Empty when the claim is ungrounded.",
    )


class CriticVerdict(BaseModel):
    """Result of checking a draft. `passed` is computed in Python, never by the model —
    the LLM supplies evidence, the code makes the decision."""

    claims: list[FactualClaim] = Field(default_factory=list)
    word_count: int = 0
    cta_count: int = 0
    banned_phrase_hits: list[str] = Field(default_factory=list)
    passed: bool = False
    critique: str = ""

    @property
    def grounded_claims(self) -> list[FactualClaim]:
        return [c for c in self.claims if c.grounded]

    @property
    def ungrounded_claims(self) -> list[FactualClaim]:
        return [c for c in self.claims if not c.grounded]
