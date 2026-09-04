"""Runs the golden lead set through enrichment + drafting and scores the output.

Never sends. Usage:
    python -m eval.runner                 # full golden set
    python -m eval.runner --limit 5       # quick pass while iterating
    python -m eval.runner --no-judge      # deterministic checks only, no LLM cost
"""

import argparse
import csv
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agents.drafter import draft_email
from agents.enricher import enrich_lead
from eval import checks, report
from eval.llm_judge import judge_draft
from eval.prompt_version import PROMPT_VERSION

GOLDEN_FILE = Path("data/golden_leads.csv")


def load_golden(limit: int | None = None) -> list[dict]:
    with open(GOLDEN_FILE, newline="", encoding="utf-8") as f:
        leads = list(csv.DictReader(f))
    return leads[:limit] if limit else leads


def _with_retry(fn, attempts: int = 3):
    """Groq rate limits are common enough that without this a run silently loses part of
    its sample, which would make two runs incomparable — the one thing the harness exists
    to prevent. Only retries rate limits; real errors surface immediately."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if "429" not in str(e) or attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def score_lead(lead: dict, use_judge: bool = True) -> dict:
    result = {"name": lead.get("name", ""), "company": lead.get("company", "")}
    started = time.time()

    try:
        research = _with_retry(lambda: enrich_lead(lead))
        draft = _with_retry(lambda: draft_email(lead, research))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["latency_s"] = round(time.time() - started, 1)
        return result

    result.update(checks.run_all(draft))
    result["draft"] = draft
    result["research"] = research

    if use_judge:
        try:
            verdict = _with_retry(lambda: judge_draft(lead, research, draft))
            result["judge_score"] = verdict.score
            result["judge_reason"] = verdict.reason
        except Exception as e:
            # A judge failure shouldn't discard a draft that scored fine on the free checks.
            result["judge_score"] = None
            result["judge_reason"] = f"judge failed: {type(e).__name__}: {e}"

    result["latency_s"] = round(time.time() - started, 1)
    return result


def run(limit: int | None = None, use_judge: bool = True):
    leads = load_golden(limit)
    print(f"Running {len(leads)} golden leads | prompt {PROMPT_VERSION} | judge={use_judge}")

    results = []
    for i, lead in enumerate(leads, 1):
        print(f"  [{i}/{len(leads)}] {lead.get('name')} @ {lead.get('company')}...")
        results.append(score_lead(lead, use_judge=use_judge))

    summary = report.summarise(results, PROMPT_VERSION)
    report.print_report(results, summary)
    report.append_history(summary)
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval harness for the outreach pipeline")
    parser.add_argument("--limit", type=int, help="Only run the first N golden leads")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge")
    args = parser.parse_args()

    run(limit=args.limit, use_judge=not args.no_judge)
