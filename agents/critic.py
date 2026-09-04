"""Quality gate between drafting and sending.

Checks a draft against the rubric and returns a verdict the pipeline can act on. The LLM's
only job here is evidence-gathering — extracting the claims an email makes and pointing at
the source text that supports each one. Whether the draft *passes* is decided in Python from
that evidence, so the gate stays deterministic and can't be talked out of a rejection.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from agents.schemas import CriticVerdict, FactualClaim
from config import CRITIC_MODEL
from eval import checks

SYSTEM_PROMPT = """You audit cold emails for factual grounding.

You are given the search results a writer was working from, and the email they produced.
Extract every factual claim the email makes ABOUT THE RECIPIENT'S COMPANY OR THE RECIPIENT,
and decide whether the search results support it.

What counts as a claim to extract:
- Statements about the recipient's company: funding, launches, growth, hiring, expansion,
  partnerships, metrics, strategy.
- Statements about the recipient personally: their role, tenure, remit, public statements.

What to IGNORE completely (these are not claims about the recipient):
- Anything about the SENDER or the sender's own company and what its product does.
- Generic value propositions, offers, opinions, flattery, and predictions.
- Questions, greetings, sign-offs, and calls to action.

Grounding rules — apply strictly:
- grounded = true ONLY if the search results actually state or directly support the claim.
  You must fill source_quote with text copied VERBATIM from the search results.
- If you cannot copy a supporting quote, the claim is NOT grounded. Do not paraphrase,
  do not reason from general knowledge, and do not give the writer the benefit of the doubt.
- A claim that is plausible, or that you happen to know is true, is still UNGROUNDED if the
  provided search results do not say it. That is the whole point of this check.
- Vague nods ("your recent momentum") with nothing specific behind them are not claims;
  skip them rather than marking them grounded."""

USER_TEMPLATE = """Search results the writer was given:
---
{sources}
---

The email they wrote:
---
{draft}
---

Extract and audit the claims."""


_STOPWORDS = {
    "about", "after", "again", "against", "their", "there", "these", "those", "which",
    "while", "would", "could", "should", "being", "before", "between", "during", "where",
    "with", "from", "that", "this", "have", "has", "had", "was", "were", "will", "into",
    "than", "then", "they", "them", "its", "his", "her", "our", "your", "and", "for",
    "the", "are", "not", "but", "also", "more", "most", "such", "over", "under", "been",
}

# How much of a quote's distinctive vocabulary must appear in the sources. Not 100%:
# models routinely drop a word or normalise a spelling when copying. Not much lower
# either — the fabrication-catching power lives entirely in this threshold.
_SUPPORT_THRESHOLD = 0.9


def _normalise(text: str) -> str:
    """Fold the typographic variation that makes literal matching fail: smart quotes,
    the several dash and space characters news sites use, and collapsed whitespace."""
    lowered = text.lower()
    for fancy, plain in (
        ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
        ("—", "-"), ("–", "-"), ("‑", "-"), ("−", "-"),
        (" ", " "), (" ", " "), (" ", " "),
    ):
        lowered = lowered.replace(fancy, plain)
    return " ".join(lowered.split())


def _distinctive_tokens(text: str) -> list[str]:
    """Words that carry identifying weight — names, numbers, figures. These are what a
    fabricated quote cannot supply, whereas filler words are free to invent."""
    tokens = []
    for raw in _normalise(text).replace("/", " ").split():
        token = raw.strip(".,;:!?()[]{}\"'`-–—")
        if not token or token in _STOPWORDS:
            continue
        if any(ch.isdigit() for ch in token) or len(token) >= 5:
            tokens.append(token)
    return tokens


def _quote_supported(quote: str, haystack: str) -> bool:
    """Is this quote really backed by the sources?

    A plain substring test is too brittle: models routinely assemble a "quote" from several
    disjoint fragments of the same article — headline plus dateline plus a sentence from the
    body — so every piece is genuine but the concatenation appears nowhere contiguously.
    Rejecting those would flag correct drafts and make the gate noise.

    So: accept a contiguous match, otherwise require nearly all of the quote's distinctive
    vocabulary to be present in the sources. Invented specifics — a funding round, a city,
    a headcount that was never reported — fail this immediately, because the numbers and
    proper nouns carrying the claim simply aren't in the text.
    """
    quote = (quote or "").strip()
    if not quote:
        return False

    if _normalise(quote) in _normalise(haystack):
        return True

    tokens = _distinctive_tokens(quote)
    if not tokens:
        return False

    hay_tokens = set(_distinctive_tokens(haystack))
    found = sum(1 for t in tokens if t in hay_tokens)
    return (found / len(tokens)) >= _SUPPORT_THRESHOLD


class _ClaimExtraction(BaseModel):
    """Wrapper because structured output needs an object at the top level, not a bare list."""

    claims: list[FactualClaim] = Field(default_factory=list)


_extractor = None


def _build_extractor():
    llm = ChatGroq(model=CRITIC_MODEL, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])
    return prompt | llm.with_structured_output(_ClaimExtraction)


def extract_claims(draft: str, sources: list[str]) -> list[FactualClaim]:
    """Extraction and verification happen in one call rather than one call per claim.

    Every model on this Groq account shares an 8000 TPM ceiling, and the sources have to be
    resent with each request, so per-claim verification would multiply the most expensive
    part of the payload by the number of claims for no extra rigour. Requiring a verbatim
    source_quote is what forces the model to actually look, rather than rubber-stamp.
    """
    global _extractor
    if _extractor is None:
        _extractor = _build_extractor()

    result = _extractor.invoke({"sources": "\n\n".join(sources), "draft": draft})
    claims = result.claims if result else []

    # Trust but verify: re-check every quote against the sources in code, so a claim only
    # counts as grounded if the evidence for it is really there.
    haystack = "\n\n".join(sources)
    for claim in claims:
        if claim.grounded and not _quote_supported(claim.source_quote, haystack):
            claim.grounded = False
    return claims


def _lead_record_source(lead: dict) -> str:
    """The lead row is itself evidence, and needs to be treated as such.

    Without it the critic flags "Priya is Head of Sales at Razorpay" as unsupported, because
    that fact came from the CRM rather than from a news article. Those are the identity facts
    every cold email opens on, so leaving them out made the gate reject correct drafts.
    """
    name = (lead.get("name") or "").strip()
    role = (lead.get("role") or "").strip()
    company = (lead.get("company") or "").strip()
    parts = [p for p in (f"{name} is {role}" if role else name, f"at {company}" if company else "") if p]
    return "Lead record (supplied by the sales team, treat as verified): " + " ".join(parts) + "."


def score_draft(draft: str, sources: list[str], lead: dict | None = None) -> CriticVerdict:
    """Full rubric check. `passed` requires zero ungrounded claims AND all format checks."""
    deterministic = checks.run_all(draft)

    # Track research separately from the lead record: adding the latter must not disguise
    # a lead we found no research for, which is a different situation the critique calls out.
    has_research = bool(sources)
    checkable = list(sources) + ([_lead_record_source(lead)] if lead else [])

    claims = extract_claims(draft, checkable) if checkable else []

    verdict = CriticVerdict(
        claims=claims,
        word_count=deterministic["word_count"],
        cta_count=deterministic["cta_count"],
        banned_phrase_hits=deterministic["cliche_hits"],
    )

    problems = []
    for claim in verdict.ungrounded_claims:
        problems.append(
            f'- The claim "{claim.text}" is not supported by the research. Remove it or '
            f"replace it with something the research actually says."
        )
    if not has_research:
        problems.append(
            "- There was no research available for this lead, so the email must not assert "
            "anything specific about the company. Keep it honest and general, or skip the lead."
        )
    if not deterministic["word_count_pass"]:
        problems.append(
            f"- The body is {verdict.word_count} words; the limit is {checks.MAX_WORDS}. Cut it down."
        )
    if not deterministic["subject_pass"]:
        problems.append('- The subject line is missing. Start the email with "Subject: ...".')
    if verdict.cta_count == 0:
        problems.append("- There is no clear call to action. Add exactly one.")
    elif verdict.cta_count > 1:
        problems.append(
            f"- There are {verdict.cta_count} asks in the email. Keep exactly one, cut the rest."
        )
    for phrase in verdict.banned_phrase_hits:
        problems.append(f'- Remove the banned phrase "{phrase}".')

    verdict.critique = "\n".join(problems)
    verdict.passed = not problems
    return verdict
