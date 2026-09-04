"""Tests for the grounding gate.

`_quote_supported` is the load-bearing function in this project: it decides whether a claim
counts as evidenced, and therefore whether an email is allowed to reach a person. It has two
failure directions and both are costly — too strict and it rejects correct drafts until the
gate is ignored as noise; too loose and fabrications sail through, which is the whole thing
this project is trying to prevent. So both directions are tested explicitly.
"""

import pytest

from agents import critic
from agents.critic import _distinctive_tokens, _lead_record_source, _normalise, _quote_supported
from agents.schemas import CriticVerdict, FactualClaim

SOURCE = (
    "Zepto elevates Roohi Raj to Vice President · Mon, 15 Jun 2026 12:50:25 GMT\n"
    "# Zepto elevates Roohi Raj to Vice President\n"
    "by MN4U Bureau\n\nJune 15, 2026\n\nin Marketing\n\n"
    "Mumbai: Quick commerce platform Zepto has promoted Roohi Raj to the role of "
    "Vice President, effective immediately."
)


class TestQuoteSupportedAcceptsRealEvidence:
    def test_contiguous_quote(self):
        assert _quote_supported("Zepto has promoted Roohi Raj to the role of Vice President", SOURCE)

    def test_quote_stitched_from_disjoint_fragments(self):
        """Models assemble quotes from headline + dateline + body. Each piece is genuine,
        but the concatenation appears nowhere contiguously — this must still pass."""
        stitched = ("Zepto elevates Roohi Raj to Vice President\n\nJune 15, 2026\n\n"
                    "Roohi Raj has been promoted to the role of Vice President at Zepto")
        assert _quote_supported(stitched, SOURCE)

    def test_typographic_variants(self):
        """Smart quotes, en/em dashes and non-breaking spaces must not break matching."""
        assert _quote_supported("Zepto elevates Roohi Raj to Vice President — June 15, 2026", SOURCE)
        assert _quote_supported("Zepto has promoted Roohi Raj", SOURCE)

    def test_case_insensitive(self):
        assert _quote_supported("ZEPTO HAS PROMOTED ROOHI RAJ", SOURCE)


class TestQuoteSupportedRejectsFabrication:
    @pytest.mark.parametrize("fabricated", [
        "Zepto closed a $400M Series F at a $12B valuation last month",
        "Zepto opened a new Berlin engineering office with 2000 hires",
        "Zepto elevates Roohi Raj to Chief Executive Officer",
        "Roohi Raj resigned from Zepto in July 2026",
    ])
    def test_invented_specifics_are_rejected(self, fabricated):
        assert not _quote_supported(fabricated, SOURCE)

    def test_empty_quote_is_never_support(self):
        assert not _quote_supported("", SOURCE)
        assert not _quote_supported("   ", SOURCE)
        assert not _quote_supported(None, SOURCE)

    def test_filler_only_quote_is_rejected(self):
        """A quote made only of stopwords carries no identifying weight, so it cannot
        evidence anything even though its words all appear in the source."""
        assert not _quote_supported("the and of to with", SOURCE)


class TestNormalisation:
    def test_collapses_whitespace_and_folds_punctuation(self):
        assert _normalise("A  B\n\nC") == "a b c"
        assert _normalise("don’t “quote” me") == "don't \"quote\" me"
        assert _normalise("a — b – c ‑ d") == "a - b - c - d"

    def test_distinctive_tokens_drop_stopwords_and_short_words(self):
        tokens = _distinctive_tokens("The company raised $400M with their new team")
        assert "400m" in tokens
        assert "company" in tokens
        assert "the" not in tokens
        assert "with" not in tokens

    def test_numbers_are_always_distinctive_even_when_short(self):
        assert "34.5%" in " ".join(_distinctive_tokens("revenue grew 34.5% this year"))


class TestLeadRecordAsSource:
    def test_lead_identity_is_treated_as_evidence(self, lead):
        record = _lead_record_source(lead)
        assert "Priya Sharma" in record
        assert "Head of Sales" in record
        assert "Razorpay" in record

    def test_claim_about_the_lead_role_is_grounded_by_the_record(self, lead):
        """Regression: the gate used to reject "Priya is Head of Sales at Razorpay" as
        unsupported, because it came from the CRM row rather than from a news article."""
        assert _quote_supported("Priya Sharma is Head of Sales at Razorpay",
                                _lead_record_source(lead))

    def test_tolerates_missing_fields(self):
        assert _lead_record_source({"name": "Someone"})
        assert _lead_record_source({})


class TestScoreDraftDecision:
    """`passed` must be computed in Python from evidence, never asserted by the model."""

    def _patch_claims(self, monkeypatch, claims):
        monkeypatch.setattr(critic, "extract_claims", lambda draft, sources: claims)

    def test_passes_when_grounded_and_well_formed(self, monkeypatch, lead):
        self._patch_claims(monkeypatch, [FactualClaim(text="c", grounded=True, source_quote="q")])
        draft = "Subject: Hi\n\nYour launch was notable. Worth a 15-min call this week?"
        verdict = critic.score_draft(draft, ["some source"], lead=lead)
        assert verdict.passed
        assert verdict.critique == ""

    def test_fails_on_any_ungrounded_claim(self, monkeypatch, lead):
        self._patch_claims(monkeypatch, [
            FactualClaim(text="real", grounded=True, source_quote="q"),
            FactualClaim(text="invented Series F", grounded=False),
        ])
        draft = "Subject: Hi\n\nYour Series F was notable. Worth a 15-min call?"
        verdict = critic.score_draft(draft, ["some source"], lead=lead)
        assert not verdict.passed
        assert "invented Series F" in verdict.critique
        assert len(verdict.grounded_claims) == 1
        assert len(verdict.ungrounded_claims) == 1

    def test_fails_when_over_word_limit(self, monkeypatch, lead):
        self._patch_claims(monkeypatch, [])
        draft = "Subject: Hi\n\n" + ("word " * 200) + "Worth a call?"
        verdict = critic.score_draft(draft, ["s"], lead=lead)
        assert not verdict.passed
        assert "limit is 120" in verdict.critique

    def test_fails_when_subject_missing(self, monkeypatch, lead):
        self._patch_claims(monkeypatch, [])
        verdict = critic.score_draft("No subject line here. Worth a call?", ["s"], lead=lead)
        assert not verdict.passed
        assert "subject line is missing" in verdict.critique.lower()

    @pytest.mark.parametrize("body,fragment", [
        ("Subject: Hi\n\nNo ask at all.", "no clear call to action"),
        ("Subject: Hi\n\nCall? Or email? Or meet?", "3 asks"),
    ])
    def test_cta_rules(self, monkeypatch, lead, body, fragment):
        self._patch_claims(monkeypatch, [])
        verdict = critic.score_draft(body, ["s"], lead=lead)
        assert not verdict.passed
        assert fragment in verdict.critique.lower()

    def test_fails_on_banned_phrase(self, monkeypatch, lead):
        self._patch_claims(monkeypatch, [])
        draft = "Subject: Hi\n\nI hope this email finds you well. Worth a call?"
        verdict = critic.score_draft(draft, ["s"], lead=lead)
        assert not verdict.passed
        assert "banned phrase" in verdict.critique.lower()

    def test_no_research_is_called_out_separately(self, monkeypatch, lead):
        """A lead with no research is a different situation from a lead with bad claims,
        and supplying the lead record must not disguise it."""
        self._patch_claims(monkeypatch, [])
        verdict = critic.score_draft("Subject: Hi\n\nHello. Worth a call?", [], lead=lead)
        assert not verdict.passed
        assert "no research available" in verdict.critique.lower()


class TestModelClaimsAreReverified:
    def test_model_asserted_grounding_is_overridden_when_quote_is_absent(self, monkeypatch):
        """If the model says grounded=True but its quote isn't in the sources, it produced
        the evidence rather than found it — the code must overrule it."""
        fake = type("R", (), {"claims": [
            FactualClaim(text="invented", grounded=True, source_quote="a $400M Series F round"),
            FactualClaim(text="real", grounded=True,
                         source_quote="Zepto has promoted Roohi Raj to the role of Vice President"),
        ]})()
        monkeypatch.setattr(critic, "_extractor", type("E", (), {"invoke": lambda self, p: fake})())

        claims = critic.extract_claims("draft text", [SOURCE])
        assert claims[0].grounded is False
        assert claims[1].grounded is True


def test_verdict_helpers_partition_claims():
    verdict = CriticVerdict(claims=[
        FactualClaim(text="a", grounded=True, source_quote="q"),
        FactualClaim(text="b", grounded=False),
        FactualClaim(text="c", grounded=True, source_quote="q"),
    ])
    assert len(verdict.grounded_claims) == 2
    assert len(verdict.ungrounded_claims) == 1
