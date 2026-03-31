import os
from dotenv import load_dotenv
load_dotenv()  

from langchain_tavily import TavilySearch  
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = """You are a B2B sales researcher. Find 2-3 specific, recent facts 
about a company and person useful for personalising a cold email.
Focus ONLY on news from 2025-2026: funding rounds, product launches, hiring, expansions.
If you only have old information, explicitly say so — do not present old news as recent.
Return ONLY a concise 3-5 sentence brief. No bullet points. No fluff."""

def enrich_lead(lead: dict) -> str:
    search = TavilySearch(max_results=3,days=30, topic="news", search_depth="advanced")
    query = f"{lead.get('company')} {lead.get('role')} recent news 2026"
    results = search.invoke(query)

    context = "\n".join([r["content"] for r in results if "content" in r])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
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