"""Tests for the draft → critique → rewrite loop.

The loop decides how many times we pay for a model call and whether an email is allowed to
be sent, so its control flow is worth pinning precisely: it must stop early on success,
retry with the critique on failure, and give up rather than loop forever.
"""

from agents import pipeline
from agents.schemas import CriticVerdict, FactualClaim


def _verdict(passed, critique=""):
    return CriticVerdict(passed=passed, critique=critique,
                         claims=[FactualClaim(text="c", grounded=passed, source_quote="q")])


def _patch(monkeypatch, verdicts, drafts=("draft-1", "draft-2", "draft-3")):
    calls = {"draft": 0, "rewrite": 0, "rewrite_args": []}
    drafts = list(drafts)
    verdicts = list(verdicts)

    def fake_draft(lead, research):
        calls["draft"] += 1
        return drafts.pop(0)

    def fake_rewrite(lead, research, previous_draft, critique):
        calls["rewrite"] += 1
        calls["rewrite_args"].append({"previous_draft": previous_draft, "critique": critique})
        return drafts.pop(0)

    monkeypatch.setattr(pipeline, "draft_email", fake_draft)
    monkeypatch.setattr(pipeline, "rewrite_email", fake_rewrite)
    monkeypatch.setattr(pipeline, "score_draft",
                        lambda draft, sources, lead=None: verdicts.pop(0))
    return calls


def test_clean_draft_is_not_rewritten(monkeypatch, lead):
    """A passing draft must cost exactly one model call — no speculative second attempt."""
    calls = _patch(monkeypatch, [_verdict(True)])

    result = pipeline.draft_with_gate(lead, "research", ["src"])

    assert result.approved
    assert result.status == "APPROVED"
    assert result.attempts == 1
    assert calls["rewrite"] == 0


def test_failed_draft_is_rewritten_and_can_pass(monkeypatch, lead):
    calls = _patch(monkeypatch, [_verdict(False, "- fix the invented claim"), _verdict(True)])

    result = pipeline.draft_with_gate(lead, "research", ["src"])

    assert result.approved
    assert result.attempts == 2
    assert calls["rewrite"] == 1
    assert result.draft == "draft-2"


def test_critique_and_previous_draft_are_passed_to_the_rewriter(monkeypatch, lead):
    """The rewrite is only useful if it actually receives the objections."""
    calls = _patch(monkeypatch, [_verdict(False, "- the Series F claim is unsupported"),
                                 _verdict(True)])

    pipeline.draft_with_gate(lead, "research", ["src"])

    args = calls["rewrite_args"][0]
    assert args["critique"] == "- the Series F claim is unsupported"
    assert args["previous_draft"] == "draft-1"


def test_gives_up_after_max_attempts_and_flags_for_review(monkeypatch, lead):
    calls = _patch(monkeypatch, [_verdict(False, "bad"), _verdict(False, "still bad")])

    result = pipeline.draft_with_gate(lead, "research", ["src"], max_attempts=2)

    assert not result.approved
    assert result.status == "NEEDS_REVIEW"
    assert result.attempts == 2
    assert calls["rewrite"] == 1


def test_rejected_draft_is_returned_not_discarded(monkeypatch, lead):
    """A human reviewing a rejected draft plus its critique beats an empty result."""
    _patch(monkeypatch, [_verdict(False, "bad"), _verdict(False, "still bad")])

    result = pipeline.draft_with_gate(lead, "research", ["src"], max_attempts=2)

    assert result.draft == "draft-2"
    assert result.verdict.critique == "still bad"


def test_max_attempts_is_configurable(monkeypatch, lead):
    calls = _patch(monkeypatch, [_verdict(False, "a"), _verdict(False, "b"), _verdict(True)],
                   drafts=["d1", "d2", "d3"])

    result = pipeline.draft_with_gate(lead, "research", ["src"], max_attempts=3)

    assert result.approved
    assert result.attempts == 3
    assert calls["rewrite"] == 2


def test_single_attempt_never_rewrites(monkeypatch, lead):
    calls = _patch(monkeypatch, [_verdict(False, "bad")])

    result = pipeline.draft_with_gate(lead, "research", ["src"], max_attempts=1)

    assert not result.approved
    assert calls["rewrite"] == 0
