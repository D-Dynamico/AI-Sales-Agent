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

def enrich_lead(lead: dict) -> str:
    search = TavilySearch(max_results=3,days=30, topic="news", search_depth="advanced")
    query = f"{lead.get('company')} {lead.get('role')} recent news 2026"
    results = search.invoke(query)

    context = "\n".join([r["content"] for r in results if "content" in r])

    llm = ChatGroq(
        model=PIPELINE_MODEL,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Lead: {name}, {role} at {company}\n\nSearch results:\n{context}\n\nWrite the research brief.")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "name": lead.get("name", ""),
        "role": lead.get("role", ""),
        "company": lead.get("company", ""),
        "context": context[:3000],
    })