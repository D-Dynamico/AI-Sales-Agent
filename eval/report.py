"""Aggregates per-lead eval results and appends a row to eval/history.csv."""

import csv
from datetime import datetime, timezone
from pathlib import Path

from config import JUDGE_MODEL, PIPELINE_MODEL

HISTORY_FILE = Path("eval/history.csv")

FIELDNAMES = [
    "timestamp",
    "prompt_version",
    "pipeline_model",
    "judge_model",
    "n_leads",
    "n_errors",
    "word_count_pass_rate",
    "subject_pass_rate",
    "cliche_pass_rate",
    "cta_pass_rate",
    "judge_mean",
    "grounded_claim_rate",
    "gate_pass_rate",
    "mean_claims_per_draft",
    "mean_attempts",
    "mean_latency_s",
]


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 3)


def summarise(results: list[dict], prompt_version: str) -> dict:
    """results: per-lead dicts from the runner. Failed leads carry 'error' and are
    excluded from quality rates — a draft that never existed can't be scored."""
    scored = [r for r in results if not r.get("error")]
    judged = [r["judge_score"] for r in scored if r.get("judge_score") is not None]
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    attempts = [r["attempts"] for r in scored if r.get("attempts") is not None]
    total_claims = sum(r.get("n_claims", 0) for r in scored)
    total_grounded = sum(r.get("n_grounded", 0) for r in scored)

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompt_version": prompt_version,
        "pipeline_model": PIPELINE_MODEL,
        "judge_model": JUDGE_MODEL,
        "n_leads": len(results),
        "n_errors": len(results) - len(scored),
        "word_count_pass_rate": _rate(scored, "word_count_pass"),
        "subject_pass_rate": _rate(scored, "subject_pass"),
        "cliche_pass_rate": _rate(scored, "cliche_pass"),
        "cta_pass_rate": _rate(scored, "cta_pass"),
        "judge_mean": round(sum(judged) / len(judged), 2) if judged else "",
        # Claim-weighted, not draft-averaged: a draft making six claims with one bad should
        # weigh more heavily than a draft making one. This is the number Phase 1 exists to move.
        "grounded_claim_rate": round(total_grounded / total_claims, 3) if total_claims else "",
        "gate_pass_rate": _rate(scored, "gate_passed"),
        "mean_claims_per_draft": round(total_claims / len(scored), 1) if scored else "",
        "mean_attempts": round(sum(attempts) / len(attempts), 2) if attempts else "",
        "mean_latency_s": round(sum(latencies) / len(latencies), 1) if latencies else "",
    }


def append_history(summary: dict):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(summary)


def print_report(results: list[dict], summary: dict):
    print("\n" + "=" * 72)
    print(f"EVAL REPORT — prompt {summary['prompt_version']} — {summary['n_leads']} leads")
    print(f"pipeline: {summary['pipeline_model']}  |  judge: {summary['judge_model']}")
    print("=" * 72)

    print(f"\n{'lead':<26} {'words':>5} {'subj':>5} {'cta':>4} "
          f"{'grounded':>9} {'att':>4} {'gate':>13} {'judge':>6}")
    print("-" * 84)
    for r in results:
        label = f"{r['name']} @ {r['company']}"[:25]
        if r.get("error"):
            print(f"{label:<26} {'— ' + r['error'][:50]}")
            continue
        grounded = f"{r.get('n_grounded', 0)}/{r.get('n_claims', 0)}"
        print(
            f"{label:<26} {r['word_count']:>5} {'ok' if r['subject_pass'] else 'MISS':>5} "
            f"{r['cta_count']:>4} {grounded:>9} {r.get('attempts', '—'):>4} "
            f"{('APPROVED' if r.get('gate_passed') else 'NEEDS_REVIEW'):>13} "
            f"{r['judge_score'] if r.get('judge_score') is not None else '—':>6}"
        )

    print("-" * 84)
    print(f"  word count <= 120 : {summary['word_count_pass_rate']:.0%}")
    print(f"  subject present   : {summary['subject_pass_rate']:.0%}")
    print(f"  no banned phrases : {summary['cliche_pass_rate']:.0%}")
    print(f"  exactly one CTA   : {summary['cta_pass_rate']:.0%}")
    print(f"  judge mean (1-5)  : {summary['judge_mean']}")
    grounded = summary["grounded_claim_rate"]
    print(f"  GROUNDED CLAIMS   : {grounded:.0%}" if grounded != "" else "  GROUNDED CLAIMS   : n/a")
    print(f"  gate pass rate    : {summary['gate_pass_rate']:.0%}")
    print(f"  claims per draft  : {summary['mean_claims_per_draft']}")
    print(f"  attempts per lead : {summary['mean_attempts']}")
    print(f"  errors            : {summary['n_errors']}")
    print(f"  mean latency      : {summary['mean_latency_s']}s per lead")
    print(f"\nAppended to {HISTORY_FILE}\n")
