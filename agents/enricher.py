import os
from dotenv import load_dotenv
load_dotenv()  

from langchain_tavily import TavilySearch  
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import PIPELINE_MODEL

# Phrased as "summarise what you are given", never "find" or "search" — the web search has
# already happened in Tavily by the time this prompt runs, and research-flavoured wording
# makes tool-capable models try to call a search tool of their own, which the API rejects.
SYSTEM_PROMPT = """You summarise pre-fetched search results for a B2B sales team.

You have no tools and no ability to search. Work only from the search results provided
in the message. Do not attempt to look anything up.

From those results, pick the 2-3 most specific and most recent facts about the company or
person that would be useful for personalising a cold email — funding rounds, product
launches, hiring, expansions. Prefer 2025-2026 items.
If the results contain only old or irrelevant information, say so plainly — never present
old news as recent, and never invent a fact that is not in the results.
Return ONLY a concise 3-5 sentence brief. No bullet points. No fluff."""

NO_RESEARCH = "No recent search results were found for this company."

MAX_CONTEXT_CHARS = 4000


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


def enrich_lead(lead: dict) -> tuple[str, list[str]]:
    """Returns (research_brief, raw_sources).

    The raw sources come back alongside the prose so the critic can check the drafted
    email's claims against what the search actually said. A brief alone is uncheckable.
    """
    search = TavilySearch(max_results=3,days=30, topic="news", search_depth="advanced")
    query = f"{lead.get('company')} {lead.get('role')} recent news 2026"
    results = search.invoke(query)

    sources = _extract_sources(results)
    if not sources:
        # Don't hand an empty context to the model — with nothing to summarise it will
        # happily produce a confident brief out of training data, which is the exact
        # failure this pipeline is supposed to prevent.
        return NO_RESEARCH, []

    context = "\n\n".join(sources)

    llm = ChatGroq(
        model=PIPELINE_MODEL,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Lead: {name}, {role} at {company}\n\nSearch results:\n{context}\n\nWrite the research brief.")
    ])

    chain = prompt | llm | StrOutputParser()

    brief = chain.invoke({
        "name": lead.get("name", ""),
        "role": lead.get("role", ""),
        "company": lead.get("company", ""),
        "context": context[:MAX_CONTEXT_CHARS],
    })

    return brief, sources