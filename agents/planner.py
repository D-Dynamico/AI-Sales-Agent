"""Decides what to look up for a lead, then runs those lookups in parallel.

The planner is rule-based, not an LLM call. The plan allowed for either; rules won for the
same reason as in `icp.py` — this step exists to make research better *per token*, and
spending a model call to decide what to search would eat the budget the extra searches need.
Rules are also easier to eval: a lookup plan is either sensible for the role or it isn't,
and you can read it without running anything.

Tavily calls are I/O-bound and independent, so they run concurrently. That keeps a
three-angle plan roughly as fast as the old single query, rather than three times slower.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime

DEFAULT_MAX_QUERIES = 3
DEFAULT_BUDGET_SECONDS = 25.0


@dataclass(frozen=True)
class SearchQuery:
    angle: str
    query: str


# Angle → the words that make a search find it. Ordered by general usefulness for cold
# outreach: a funding round or launch is a stronger opener than a generic news sweep.
_ANGLE_TERMS = {
    "funding": "funding round investment raise valuation",
    "product_launch": "launch launches new product announcement",
    "hiring": "hiring expands team appoints promotes leadership",
    "expansion": "expansion new market opens office growth",
    "financials": "revenue results quarterly earnings profit loss",
    "partnership": "partnership partners integration deal",
}

# Which angles matter for which kind of role. A VP Marketing cares that the company shipped
# something; a CTO cares what it was built on; a founder cares about money and momentum.
_ROLE_ANGLES = {
    ("sales", "revenue", "growth", "gtm", "outbound", "commercial", "bizdev"):
        ["funding", "hiring", "expansion"],
    ("marketing", "brand", "demand", "content"):
        ["product_launch", "funding", "partnership"],
    ("engineering", "technology", "cto", "technical", "product", "data", "platform"):
        ["product_launch", "hiring", "partnership"],
    ("operations", "ops", "supply", "logistics"):
        ["expansion", "financials", "hiring"],
    ("founder", "ceo", "chief", "president", "strategy", "staff"):
        ["funding", "financials", "expansion"],
}

_DEFAULT_ANGLES = ["funding", "product_launch", "hiring"]


def plan_lookups(lead: dict, max_queries: int = DEFAULT_MAX_QUERIES) -> list[SearchQuery]:
    """Pick the lookup angles worth spending a search on for this lead."""
    company = (lead.get("company") or "").strip()
    if not company:
        return []

    role_text = (lead.get("role") or "").lower()
    angles = None
    for keywords, mapped in _ROLE_ANGLES.items():
        if any(word in role_text for word in keywords):
            angles = list(mapped)
            break
    if angles is None:
        angles = list(_DEFAULT_ANGLES)

    # "recent news <year>" is load-bearing, not decoration. Dropping it (as the first
    # version of this planner did) surfaced a 2021 funding article for a 2026 query even
    # with Tavily's days=30 filter set — keyword-only queries pull in evergreen coverage,
    # and the enricher then correctly reports low confidence and finds nothing usable.
    year = datetime.now().year
    return [
        SearchQuery(angle=angle, query=f"{company} {_ANGLE_TERMS[angle]} recent news {year}")
        for angle in angles[:max_queries]
    ]


async def _run_one(search, query: SearchQuery):
    """TavilySearch is sync, so it goes to a worker thread — the point is overlapping the
    network waits, and to_thread achieves that without depending on library async support."""
    return query, await asyncio.to_thread(search.invoke, query.query)


async def _gather(search, queries: list[SearchQuery], budget_seconds: float):
    tasks = [asyncio.create_task(_run_one(search, q)) for q in queries]
    done, pending = await asyncio.wait(tasks, timeout=budget_seconds)
    for task in pending:
        task.cancel()

    results = []
    for task in done:
        try:
            results.append(task.result())
        except Exception:
            # One angle failing is not a failed lead — the others still carry research.
            continue
    return results


def run_lookups(search, queries: list[SearchQuery],
                budget_seconds: float = DEFAULT_BUDGET_SECONDS) -> list[tuple[SearchQuery, dict]]:
    """Run every planned lookup concurrently, dropping any that exceed the time budget.

    Partial results are returned rather than raising: research is a best-effort input, and
    two angles out of three is materially better than nothing.
    """
    if not queries:
        return []
    return asyncio.run(_gather(search, queries, budget_seconds))


def timed_run_lookups(search, queries, budget_seconds: float = DEFAULT_BUDGET_SECONDS):
    started = time.time()
    results = run_lookups(search, queries, budget_seconds)
    return results, round(time.time() - started, 1)
