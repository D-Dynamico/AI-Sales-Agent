"""Tests for ICP scoring.

This filter decides who never gets contacted, so its false negatives are invisible by
construction — a wrongly skipped lead produces no output to notice. The tests therefore
lean on pinning which leads are *kept*, not just which are dropped.
"""

import pytest

from agents.icp import passes_icp, score_icp


class TestQualifiedLeads:
    @pytest.mark.parametrize("role", [
        "Head of Sales", "VP Marketing", "Chief Revenue Officer", "Director of Operations",
        "Co-founder", "Head of Growth", "VP Business Development", "Chief of Staff",
    ])
    def test_senior_and_relevant_roles_score_full(self, role):
        assert score_icp({"role": role, "company": "Acme", "email": "a@b.com"}).score == 1.0

    def test_senior_but_off_function_still_scores_partially(self):
        """A CTO isn't the target buyer but is worth a human's judgement, not an auto-skip."""
        result = score_icp({"role": "VP Engineering", "company": "Acme", "email": "a@b.com"})
        assert result.score == 0.5

    def test_relevant_function_without_seniority_scores_partially(self):
        result = score_icp({"role": "Sales Manager", "company": "Acme", "email": "a@b.com"})
        assert result.score == 0.5


class TestDisqualified:
    @pytest.mark.parametrize("role", ["Intern", "Sales Intern", "Student", "Recruiter"])
    def test_disqualifying_titles_score_zero(self, role):
        assert score_icp({"role": role, "company": "Acme", "email": "a@b.com"}).score == 0.0

    def test_missing_email_scores_zero(self):
        assert score_icp({"role": "Head of Sales", "company": "Acme", "email": ""}).score == 0.0

    def test_missing_company_scores_zero(self):
        assert score_icp({"role": "Head of Sales", "company": "", "email": "a@b.com"}).score == 0.0

    def test_disqualification_beats_seniority(self):
        """"Head of Recruiting" is senior and out of scope; the disqualifier must win."""
        assert score_icp({"role": "Head of Recruiter", "company": "A",
                          "email": "a@b.com"}).score == 0.0


def test_missing_role_is_weak_signal_not_a_disqualification():
    """Exported lead lists often lose the title field; that shouldn't silently bin the lead."""
    result = score_icp({"role": "", "company": "Acme", "email": "a@b.com"})
    assert result.score == 0.3
    assert "no role" in result.summary


def test_scores_are_explainable():
    """A skipped lead has to come with the reason it was skipped."""
    result = score_icp({"role": "Head of Sales", "company": "Acme", "email": "a@b.com"})
    assert "senior title" in result.summary
    assert "relevant function" in result.summary


class TestThreshold:
    def test_default_threshold_keeps_borderline_leads(self):
        """The default must drop clear mismatches only — 0.5 leads still get through."""
        ok, _ = passes_icp({"role": "VP Engineering", "company": "A", "email": "a@b.com"}, 0.3)
        assert ok

    def test_threshold_zero_disables_skipping(self):
        ok, _ = passes_icp({"role": "Intern", "company": "A", "email": "a@b.com"}, 0.0)
        assert ok

    def test_strict_threshold_drops_partial_matches(self):
        ok, _ = passes_icp({"role": "VP Engineering", "company": "A", "email": "a@b.com"}, 1.0)
        assert not ok

    def test_case_and_punctuation_insensitive(self):
        ok, _ = passes_icp({"role": "head of SALES,", "company": "A", "email": "a@b.com"}, 1.0)
        assert ok
