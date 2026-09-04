"""Structured types shared across the pipeline."""

from pydantic import BaseModel, Field


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
