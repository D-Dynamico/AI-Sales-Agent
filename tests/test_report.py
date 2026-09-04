"""Tests for eval aggregation.

The grounded-claim rate is the number this project reports to justify its prompt and model
changes, so how it's computed matters: it's claim-weighted, not draft-averaged, and failed
leads must not silently inflate it — a bias that already produced one misleading baseline
(v1 scored 4.12 by averaging only the leads that didn't error).
"""

from eval.report import summarise


def _lead(**kw):
    base = {"name": "N", "company": "C", "word_count_pass": True, "subject_pass": True,
            "cliche_pass": True, "cta_pass": True, "gate_passed": True, "judge_score": 4,
            "n_claims": 2, "n_grounded": 2, "attempts": 1, "latency_s": 10.0}
    base.update(kw)
    return base


def test_grounded_rate_is_claim_weighted_not_draft_averaged():
    """A draft making 6 claims with 1 bad should weigh more than a draft making 1 good one."""
    results = [_lead(n_claims=6, n_grounded=5), _lead(n_claims=1, n_grounded=1)]

    summary = summarise(results, "v1")

    assert summary["grounded_claim_rate"] == round(6 / 7, 3)


def test_errored_leads_are_excluded_from_quality_rates():
    """The v1 baseline overstated quality by averaging only leads that succeeded; errors
    must still be counted and reported, just not scored."""
    results = [_lead(), _lead(), {"name": "N", "company": "C", "error": "boom", "latency_s": 2.0}]

    summary = summarise(results, "v1")

    assert summary["n_leads"] == 3
    assert summary["n_errors"] == 1
    assert summary["word_count_pass_rate"] == 1.0
    assert summary["mean_claims_per_draft"] == 2.0


def test_gate_pass_rate_counts_held_drafts():
    results = [_lead(gate_passed=True), _lead(gate_passed=False), _lead(gate_passed=True)]
    assert summarise(results, "v1")["gate_pass_rate"] == 0.667


def test_records_model_identity_for_comparability():
    """Two runs on the same prompt version but different models must be distinguishable."""
    summary = summarise([_lead()], "v2")
    assert summary["pipeline_model"]
    assert summary["judge_model"]
    assert summary["prompt_version"] == "v2"


def test_mean_attempts_reflects_rewrites():
    results = [_lead(attempts=1), _lead(attempts=2), _lead(attempts=2)]
    assert summarise(results, "v1")["mean_attempts"] == 1.67


def test_handles_empty_and_all_error_runs():
    assert summarise([], "v1")["n_leads"] == 0
    all_errors = summarise([{"name": "N", "company": "C", "error": "x"}], "v1")
    assert all_errors["n_errors"] == 1
    assert all_errors["grounded_claim_rate"] == ""
    assert all_errors["judge_mean"] == ""


def test_missing_judge_scores_do_not_break_the_mean():
    """A judge failure must not discard a draft that scored fine on the free checks."""
    results = [_lead(judge_score=5), _lead(judge_score=None), _lead(judge_score=3)]
    assert summarise(results, "v1")["judge_mean"] == 4.0
