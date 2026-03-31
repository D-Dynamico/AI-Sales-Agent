import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


SYSTEM_PROMPT = """You write hyper-personalised cold emails on behalf of Alchemyst AI — 
an AI infrastructure company that helps B2B teams automate sales, marketing, and ops workflows.

Rules (non-negotiable):
1. Max 120 words. Brevity is respect.
2. Open with a specific observation about the company or person — never a generic opener.
3. Never say: "I hope this email finds you well", "I came across your profile", 
   "I wanted to reach out", "touch base", or "synergy".
4. One clear, low-friction CTA (e.g. "Worth a 15-min call this week?")
5. No bullet points. Flowing prose only.
6. Sign off as: Alex, Alchemyst AI
7. Subject line: punchy, specific, under 8 words. Include it at the top as "Subject: ..."

Tone: Direct. Intelligent. Peer-to-peer. Not salesy.
"""

USER_TEMPLATE = """Write a cold email for this lead using the research below.

Lead info:
- Name: {name}
- Role: {role}
- Company: {company}

Research brief:
{research}

Write the subject line first, then the email body.
"""


def build_drafter():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])

    return prompt | llm | StrOutputParser()


_drafter = None


def draft_email(lead: dict, research: str) -> str:
    """
    Takes a lead dict and research brief string.
    Returns the full email (subject line + body) as a string.
    """
    global _drafter
    if _drafter is None:
        _drafter = build_drafter()

    return _drafter.invoke({
        "name": lead.get("name", ""),
        "role": lead.get("role", ""),
        "company": lead.get("company", ""),
        "research": research,
    })