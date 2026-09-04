import os
from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents import cache
from agents.planner import DEFAULT_BUDGET_SECONDS, plan_lookups, run_lookups
from agents.schemas import ResearchResult, TriggerEvent
from config import ENRICHMENT_TTL_SECONDS, PIPELINE_MODEL, USE_RESEARCH_CACHE

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


_structurer = None
_json_fallback = None


def _build_structurer():
    llm = ChatGroq(model=PIPELINE_MODEL, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])
    return prompt | llm.with_structured_output(_ResearchExtraction)


def _build_json_fallback():
    """Plain-JSON escape hatch for when tool-call structured output fails.

    Not paranoia: a live run produced four perfectly quoted events and then wrote
    `"confidence": 0. nine`, which made the tool call unparseable and threw the entire
    result away. Good research shouldn't be lost to one malformed token, so on failure we
    ask again in plain text and parse it ourselves.
    """
    llm = ChatGroq(model=PIPELINE_MODEL, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + """

Reply with ONLY a JSON object, no prose and no code fences, in exactly this shape:
{{"trigger_events": [{{"kind": "funding", "summary": "...", "source_quote": "..."}}],
  "pain_hypotheses": ["..."], "confidence": 0.5}}
`confidence` must be a plain decimal number such as 0.8 — never a word."""),
        ("human", USER_TEMPLATE),
    ])
    return prompt | llm | StrOutputParser()


def _parse_json_block(text: str) -> _ResearchExtraction:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned.lstrip("`")
        cleaned = cleaned.removeprefix("json").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return _ResearchExtraction.model_validate_json(cleaned[start:end + 1])


def _is_infrastructure_error(exc: Exception) -> bool:
    """Rate limits, auth failures and outages are not 'the model said nothing useful'."""
    text = str(exc).lower()
    return any(marker in text for marker in
               ("429", "rate limit", "rate_limit", "401", "403", "authentication",
                "insufficient_quota", "503", "502", "overloaded"))


def _structure(payload: dict) -> _ResearchExtraction:
    """Structured output first, plain-JSON retry second.

    Only *parsing* failures are recoverable here. An infrastructure failure is re-raised so
    the caller logs ENRICH_ERROR: swallowing a rate limit would return "no research found"
    for a search that was never actually answered, and the pipeline would go on to write
    confident generic emails while reporting the web had nothing to say. That is the same
    silent-failure shape as the discarded-Tavily-results bug, and it must not be recreated
    in the error handling built to guard against it.
    """
    global _structurer, _json_fallback

    if _structurer is None:
        _structurer = _build_structurer()
    try:
        return _structurer.invoke(payload) or _ResearchExtraction()
    except Exception as exc:
        if _is_infrastructure_error(exc):
            raise

    if _json_fallback is None:
        _json_fallback = _build_json_fallback()
    try:
        return _parse_json_block(_json_fallback.invoke(payload))
    except Exception as exc:
        if _is_infrastructure_error(exc):
            raise
        # Both attempts produced unparseable text: the model genuinely answered, we just
        # can't read it. Empty research is the honest result, and downstream handles it.
        return _ResearchExtraction()


def research_company(lead: dict, budget_seconds: float = DEFAULT_BUDGET_SECONDS,
                     use_cache: bool = USE_RESEARCH_CACHE) -> ResearchResult:
    """Plan lookups, run them in parallel, and structure what comes back.

    Cached per company, not per lead: ten contacts at one firm share one research pass.
    """
    company = (lead.get("company") or "").strip()

    if use_cache and company:
        cached = cache.get(company, ttl_seconds=ENRICHMENT_TTL_SECONDS)
        if cached:
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
        result = ResearchResult(company=company)
        if use_cache and company:
            cache.set(company, result.model_dump())
        return result

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

    if use_cache and company:
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
