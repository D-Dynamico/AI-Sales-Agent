"""Structured LLM output with a fallback, in one place.

Extracted from the enricher rather than copied into the drafter. The logic here is small but
subtle — which failures are recoverable, which must propagate — and this project has already
shipped the same silent-failure bug twice by handling errors in two places with two slightly
different `except` clauses. One implementation, tested once.

Why a fallback exists at all: a live call produced four perfectly quoted research events and
then wrote `"confidence": 0. nine` instead of `0.9`, making the tool call unparseable and
discarding the whole result. Good output shouldn't be lost to one malformed token.
"""

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import PIPELINE_MODEL

INFRASTRUCTURE_MARKERS = (
    "429", "rate limit", "rate_limit", "401", "403", "authentication",
    "insufficient_quota", "quota", "503", "502", "overloaded", "timeout",
    # Groq's daily-cap message; matched on its own wording rather than relying on the
    # "429" prefix, since the phrasing is what actually identifies it.
    "tokens per day", "(tpd)",
)


def is_infrastructure_error(exc: Exception) -> bool:
    """Rate limits, auth failures and outages are not 'the model said nothing useful'.

    Treating them as empty output is how a 429 turns into "no research found" and the
    pipeline cheerfully writes generic emails about a question it never actually asked.
    """
    text = str(exc).lower()
    return any(marker in text for marker in INFRASTRUCTURE_MARKERS)


def parse_json_block(text: str, schema):
    """Parse a JSON object out of a plain-text reply, tolerating fences and stray prose."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned.lstrip("`")
        cleaned = cleaned.removeprefix("json").strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return schema.model_validate(json.loads(cleaned[start:end + 1]))


class StructuredChain:
    """Calls a model for structured output, falling back to plain JSON if that fails.

    `on_unparseable` decides what happens when the model answers but neither attempt can be
    read: "empty" returns a default-constructed schema (right for research, where "found
    nothing" is a legitimate result the pipeline handles honestly), "raise" propagates
    (right for drafting, where a silently empty email would be worse than a logged error).
    """

    def __init__(self, schema, system_prompt: str, user_template: str, *,
                 model: str | None = None, temperature: float = 0.2,
                 json_hint: str = "", on_unparseable: str = "empty"):
        self.schema = schema
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.model = model or PIPELINE_MODEL
        self.temperature = temperature
        self.json_hint = json_hint
        self.on_unparseable = on_unparseable
        self._primary = None
        self._fallback = None

    def _build_primary(self):
        llm = ChatGroq(model=self.model, temperature=self.temperature)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self.user_template),
        ])
        return prompt | llm.with_structured_output(self.schema)

    def _build_fallback(self):
        # Temperature 0: this is a second attempt at a format that just failed, not another
        # roll of the dice on content.
        llm = ChatGroq(model=self.model, temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\n" + (
                self.json_hint or
                "Reply with ONLY a JSON object matching the requested fields — no prose, no "
                "code fences. Numbers must be plain decimals such as 0.8, never words."
            )),
            ("human", self.user_template),
        ])
        return prompt | llm | StrOutputParser()

    def invoke(self, payload: dict):
        if self._primary is None:
            self._primary = self._build_primary()
        try:
            result = self._primary.invoke(payload)
            if result is not None:
                return result
        except Exception as exc:
            if is_infrastructure_error(exc):
                raise

        if self._fallback is None:
            self._fallback = self._build_fallback()
        try:
            return parse_json_block(self._fallback.invoke(payload), self.schema)
        except Exception as exc:
            if is_infrastructure_error(exc):
                raise
            if self.on_unparseable == "raise":
                raise ValueError(
                    f"{self.schema.__name__}: model output could not be parsed after a retry"
                ) from exc
            return self.schema()
