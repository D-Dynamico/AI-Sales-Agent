import os
from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from agents import cache
from agents.planner import DEFAULT_BUDGET_SECONDS, plan_lookups, run_lookups
from agents.schemas import ResearchResult, TriggerEvent
from agents.structured import StructuredChain
from config import ENRICHMENT_TTL_SECONDS, USE_RESEARCH_CACHE

NO_RESEARCH = "No recent search results were found for this company."

MAX_CONTEXT_CHARS = 6000

# Phrased as "summarise what you are given", never "find" or "search" — the web search has
# already happened in Tavily by the time this prompt runs, and research-flavoured wording
# makes tool-capable models try to call a search tool of their own, which the API rejects.
SYSTEM_PROMPT = """You structure pre-fetched search results for a B2B sales team.

You have no tools and no ability to search. Work only from the search results provided
in the message. Do not attempt to look anything up.

Extract the specific, recent things that happened to this company — funding, product
launches, hiring and leadership changes, expansion, partnerships, financial results.
Prefer 2025-2026 items.

Rules:
- Every trigger_event MUST include a source_quote copied VERBATIM from the search results.
  If you cannot copy a supporting quote, leave the event out entirely.
- Never invent an event, a number, or a date. Never convert or restate figures into
  different units — copy them as the source writes them.
- pain_hypotheses are your inference about problems these events imply. Keep them clearly
  speculative and few. They are not facts about the company.
- confidence: 0-1, how much genuinely recent and specific signal you found. Thin, stale or
  off-topic results should score low. Be honest — a low score is useful information."""

USER_TEMPLATE = """Lead: {name}, {role} at {company}

Search results:
{context}

Structure the research."""


def _extract_sources(results) -> list[str]:
    """Pull the usable snippets out of a Tavily response.

    langchain-tavily returns a dict shaped {'query': ..., 'results': [...], ...}; older
    versions returned a bare list. Iterating the dict directly yields its *keys*, which is
    how this function's predecessor silently passed an empty context to the LLM on every
    single lead — the search ran, was billed, and was thrown away, leaving the model to
    invent the "research" from memory. Handle both shapes so a library update can't
    reintroduce that quietly.
    """
    items = results.get("results", []) if isinstance(results, dict) else (results or [])

    sources = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        header = " · ".join(
            part for part in ((item.get("title") or "").strip(),
                              (item.get("published_date") or "").strip()) if part
        )
        url = (item.get("url") or "").strip()
        sources.append(f"{header}\n{content}\n[{url}]".strip() if header else content)
    return sources


def _dedupe(sources: list[str]) -> list[str]:
    """Angles overlap — a funding search and an expansion search often return the same
    article. Deduplicating before the LLM call keeps multi-angle research from costing
    multiples of the tokens for the same text."""
    seen, unique = set(), []
    for source in sources:
        key = " ".join(source.lower().split())[:200]
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


class _ResearchExtraction(BaseModel):
    """What the model is actually asked to produce.

    Deliberately excludes `raw_sources`: we already have those, and asking the model to echo
    them back cost hundreds of output tokens per lead and gave it one more field to
    malform. The full `ResearchResult` is assembled from this plus the sources we held.
    """

    trigger_events: list[TriggerEvent] = Field(default_factory=list)
    pain_hypotheses: list[str] = Field(default_factory=list)
    confidence: float = 0.0


_JSON_HINT = ("Reply with ONLY a JSON object of the form "
              '{{"trigger_events": [{{"kind": "funding", "summary": "...", '
              '"source_quote": "..."}}], "pain_hypotheses": ["..."], "confidence": 0.5}} '
              "- no prose, no code fences. `confidence` must be a plain decimal such as "
              "0.8, never a word.")

# on_unparseable="empty": for research, "found nothing" is a legitimate outcome the rest of
# the pipeline already handles honestly. Infrastructure errors still propagate, so a rate
# limit never masquerades as an empty web.
_structurer = StructuredChain(
    _ResearchExtraction, SYSTEM_PROMPT, USER_TEMPLATE,
    temperature=0.2, json_hint=_JSON_HINT, on_unparseable="empty",
)


def _structure(payload: dict) -> _ResearchExtraction:
    return _structurer.invoke(payload)


def research_company(lead: dict, budget_seconds: float = DEFAULT_BUDGET_SECONDS,
                     use_cache: bool = USE_RESEARCH_CACHE) -> ResearchResult:
    """Plan lookups, run them in parallel, and structure what comes back.

    Cached per company, not per lead: ten contacts at one firm share one research pass.
    """
    company = (lead.get("company") or "").strip()

    if use_cache and company:
        cached = cache.get(company, ttl_seconds=ENRICHMENT_TTL_SECONDS)
        # Only results with events are worth serving. An empty entry is indistinguishable from
        # a failed run — a search error used to be recorded as "0 sources" and then served for
        # a week, so the deployed app never searched again and never showed the error.
        if cached and cached.get("trigger_events"):
            return ResearchResult.model_validate(cached)

    queries = plan_lookups(lead)
    if not queries:
        return ResearchResult(company=company)

    search = TavilySearch(max_results=3, days=30, topic="news", search_depth="advanced")
    raw = run_lookups(search, queries, budget_seconds=budget_seconds)

    sources = _dedupe([s for _, results in raw for s in _extract_sources(results)])
    if not sources:
        # Don't hand an empty context to the model — with nothing to summarise it will
        # happily produce a confident brief out of training data, which is the exact
        # failure this pipeline is supposed to prevent.
        # Not cached: an empty search is cheap to repeat (no LLM call), and caching it would
        # pin a transient failure in place for the whole TTL.
        return ResearchResult(company=company)

    extraction = _structure({
        "name": lead.get("name", ""),
        "role": lead.get("role", ""),
        "company": company,
        "context": "\n\n".join(sources)[:MAX_CONTEXT_CHARS],
    })

    result = ResearchResult(
        company=company,
        trigger_events=extraction.trigger_events,
        pain_hypotheses=extraction.pain_hypotheses,
        confidence=extraction.confidence,
        raw_sources=sources,
    )

    # Drop events the model couldn't quote — same rule the critic applies downstream, but
    # enforced here so an unsupported "fact" never reaches the drafter in the first place.
    result.trigger_events = [e for e in result.trigger_events if (e.source_quote or "").strip()]

    # Only cache an empty extraction when there was nothing to extract from. Sources present
    # but zero events is usually a transient structuring failure, and caching it pinned the
    # company to "no research" for the full TTL even after the failure had cleared.
    if use_cache and company and result.trigger_events:
        cache.set(company, result.model_dump())
    return result


def enrich_lead(lead: dict) -> tuple[str, list[str]]:
    """Returns (research_brief, raw_sources).

    Kept as the pipeline's entry point so callers didn't all have to change when enrichment
    became structured internally. The raw sources come back alongside the prose because the
    critic checks the drafted email's claims against them — a brief alone is uncheckable.
    """
    result = research_company(lead)
    return result.to_brief(), result.raw_sources
