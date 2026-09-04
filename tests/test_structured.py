"""Tests for structured output and its fallback.

This logic was extracted from the enricher so the drafter could reuse it rather than grow a
second, subtly different copy. These tests are the reason that extraction is safe: they pin
which failures recover and which propagate, once, for every caller.
"""

import pytest
from pydantic import BaseModel

from agents.schemas import EmailDraft
from agents.structured import StructuredChain, is_infrastructure_error, parse_json_block


class Sample(BaseModel):
    value: str = ""
    score: float = 0.0


def _chain(primary=None, fallback=None, **kw):
    chain = StructuredChain(Sample, "system", "human {x}", **kw)
    if primary is not None:
        chain._primary = type("P", (), {"invoke": staticmethod(primary)})()
    if fallback is not None:
        chain._fallback = type("F", (), {"invoke": staticmethod(fallback)})()
    return chain


class TestErrorClassification:
    @pytest.mark.parametrize("message", [
        "Error code: 429 - rate limit reached",
        "tokens per day (TPD): Limit 200000",
        "Error code: 401 - authentication failed",
        "Error code: 503 - service unavailable",
        "upstream overloaded",
    ])
    def test_infrastructure_errors_are_recognised(self, message):
        assert is_infrastructure_error(RuntimeError(message))

    @pytest.mark.parametrize("message", [
        "Failed to parse tool call arguments as JSON",
        "Tool choice is none, but model called a tool",
        "validation error for Sample",
    ])
    def test_model_output_errors_are_not_infrastructure(self, message):
        assert not is_infrastructure_error(ValueError(message))


class TestFallbackBehaviour:
    def test_primary_result_is_used_when_it_works(self):
        chain = _chain(primary=lambda p: Sample(value="ok", score=1.0),
                       fallback=lambda p: pytest.fail("fallback must not run"))
        assert chain.invoke({"x": 1}).value == "ok"

    def test_unparseable_primary_falls_back_to_json(self):
        """The real case: four good events lost to `"confidence": 0. nine`."""
        chain = _chain(
            primary=lambda p: (_ for _ in ()).throw(ValueError("Failed to parse tool call")),
            fallback=lambda p: '{"value": "recovered", "score": 0.9}')
        result = chain.invoke({"x": 1})
        assert result.value == "recovered"
        assert result.score == 0.9

    def test_primary_returning_none_falls_back(self):
        chain = _chain(primary=lambda p: None,
                       fallback=lambda p: '{"value": "recovered", "score": 0.1}')
        assert chain.invoke({"x": 1}).value == "recovered"

    def test_rate_limit_propagates_without_a_second_call(self):
        """Swallowing a 429 turns "never answered" into "answered with nothing" — the
        silent-failure shape this project already shipped once."""
        chain = _chain(
            primary=lambda p: (_ for _ in ()).throw(RuntimeError("Error code: 429 rate limit")),
            fallback=lambda p: pytest.fail("must not burn another call on a rate limit"))
        with pytest.raises(RuntimeError, match="429"):
            chain.invoke({"x": 1})

    def test_rate_limit_during_fallback_also_propagates(self):
        chain = _chain(
            primary=lambda p: (_ for _ in ()).throw(ValueError("bad tool call")),
            fallback=lambda p: (_ for _ in ()).throw(RuntimeError("429 rate limit")))
        with pytest.raises(RuntimeError):
            chain.invoke({"x": 1})


class TestUnparseableHandling:
    def test_empty_mode_returns_a_default_instance(self):
        """Right for research: "found nothing" is a legitimate result downstream handles."""
        chain = _chain(primary=lambda p: (_ for _ in ()).throw(ValueError("bad")),
                       fallback=lambda p: "not json at all", on_unparseable="empty")
        result = chain.invoke({"x": 1})
        assert result.value == ""
        assert result.score == 0.0

    def test_raise_mode_surfaces_the_failure(self):
        """Right for drafting: a silently empty email would be sent as blank mail."""
        chain = _chain(primary=lambda p: (_ for _ in ()).throw(ValueError("bad")),
                       fallback=lambda p: "not json at all", on_unparseable="raise")
        with pytest.raises(ValueError, match="could not be parsed"):
            chain.invoke({"x": 1})


class TestJsonParsing:
    def test_plain_object(self):
        assert parse_json_block('{"value": "a", "score": 0.5}', Sample).score == 0.5

    def test_fenced_block(self):
        assert parse_json_block('```json\n{"value": "a", "score": 0.5}\n```', Sample).value == "a"

    def test_surrounding_prose(self):
        parsed = parse_json_block('Sure!\n{"value": "a", "score": 0.5}\nHope that helps.', Sample)
        assert parsed.value == "a"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="no JSON object"):
            parse_json_block("I can't do that", Sample)


class TestEmailDraft:
    def test_render_produces_the_classic_form(self):
        assert EmailDraft(subject="Hi there", body="Body text").render() == \
            "Subject: Hi there\n\nBody text"

    def test_subject_and_body_stay_separate(self):
        """The point of the change: no regex ever has to find the subject again."""
        draft = EmailDraft(subject="Your RAY launch", body="Congrats on RAY. Worth a call?")
        assert "Subject:" not in draft.body
        assert draft.subject == "Your RAY launch"
