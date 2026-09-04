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
        # Filled in from Phase 1 onward, once the critic exists.
        "grounded_claim_rate": "",
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

    print(f"\n{'lead':<28} {'words':>6} {'subj':>5} {'clich':>6} {'cta':>4} {'judge':>6}")
    print("-" * 72)
    for r in results:
        label = f"{r['name']} @ {r['company']}"[:27]
        if r.get("error"):
            print(f"{label:<28} {'— ' + r['error'][:38]}")
            continue
        print(
            f"{label:<28} {r['word_count']:>6} {'ok' if r['subject_pass'] else 'MISS':>5} "
            f"{'ok' if r['cliche_pass'] else 'HIT':>6} {r['cta_count']:>4} "
            f"{r['judge_score'] if r.get('judge_score') is not None else '—':>6}"
        )

    print("-" * 72)
    print(f"  word count <= 120 : {summary['word_count_pass_rate']:.0%}")
    print(f"  subject present   : {summary['subject_pass_rate']:.0%}")
    print(f"  no banned phrases : {summary['cliche_pass_rate']:.0%}")
    print(f"  exactly one CTA   : {summary['cta_pass_rate']:.0%}")
    print(f"  judge mean (1-5)  : {summary['judge_mean']}")
    print(f"  grounded claims   : {summary['grounded_claim_rate'] or 'n/a until Phase 1'}")
    print(f"  errors            : {summary['n_errors']}")
    print(f"  mean latency      : {summary['mean_latency_s']}s per lead")
    print(f"\nAppended to {HISTORY_FILE}\n")
