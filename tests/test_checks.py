"""Tests for the deterministic draft checks.

These are pure functions with no I/O, so they need no mocking and cost nothing to run.
They also back the critic's format rules, so a change here silently changes what the gate
lets through.
"""

import pytest

from agents.drafter import BANNED_PHRASES
from eval.checks import (cliche_hits, cta_count, has_subject, run_all, split_subject_body,
                         word_count)

GOOD = ("Subject: RAY's WhatsApp launch\n\n"
        "Priya, the RAY launch with IndusInd stood out. Worth a 15-min call this week?\n\n"
        "Alex, Alchemyst AI")


class TestSubjectHandling:
    def test_splits_subject_from_body(self):
        subject, body = split_subject_body(GOOD)
        assert subject == "RAY's WhatsApp launch"
        assert body.startswith("Priya,")

    def test_missing_subject(self):
        subject, body = split_subject_body("Just a body, no subject.")
        assert subject == ""
        assert body == "Just a body, no subject."
        assert not has_subject("Just a body, no subject.")

    def test_subject_detection_is_case_insensitive(self):
        assert has_subject("SUBJECT: shouting\n\nbody")
        assert has_subject("subject: quiet\n\nbody")

    def test_empty_subject_value_does_not_count(self):
        assert not has_subject("Subject:   \n\nbody")


class TestWordCount:
    def test_excludes_the_subject_line(self):
        """The 120-word budget is about the body; counting the subject would penalise
        longer subject lines that the drafting rules explicitly allow."""
        draft = "Subject: one two three four five\n\nword word word"
        assert word_count(draft) == 3

    def test_counts_body_without_subject(self):
        assert word_count("no subject here at all") == 5

    def test_realistic_draft_is_under_limit(self):
        assert word_count(GOOD) < 120


class TestClicheDetection:
    @pytest.mark.parametrize("phrase", BANNED_PHRASES)
    def test_every_banned_phrase_is_detected(self, phrase):
        """Parametrised over the shared constant, so adding a phrase to the drafter's
        prompt automatically adds a test for it — the two cannot drift apart."""
        assert cliche_hits(f"Subject: Hi\n\n{phrase} and then some.") == [phrase]

    def test_detection_is_case_insensitive(self):
        assert cliche_hits("I HOPE THIS EMAIL FINDS YOU WELL")

    def test_clean_draft_has_no_hits(self):
        assert cliche_hits(GOOD) == []


class TestCtaCount:
    def test_counts_questions_in_body(self):
        assert cta_count("Subject: Hi\n\nWorth a call?") == 1
        assert cta_count("Subject: Hi\n\nCall? Email? Meet?") == 3
        assert cta_count("Subject: Hi\n\nNo ask here.") == 0

    def test_ignores_question_marks_in_the_subject(self):
        assert cta_count("Subject: Worth a chat?\n\nNo ask in the body.") == 0


class TestRunAll:
    def test_good_draft_passes_everything(self):
        result = run_all(GOOD)
        assert result["word_count_pass"]
        assert result["subject_pass"]
        assert result["cliche_pass"]
        assert result["cta_pass"]

    def test_reports_each_failure_independently(self):
        bad = "I wanted to reach out. " + ("word " * 150) + "Call? Email?"
        result = run_all(bad)
        assert not result["subject_pass"]
        assert not result["word_count_pass"]
        assert not result["cliche_pass"]
        assert not result["cta_pass"]
        assert result["cta_count"] == 2
