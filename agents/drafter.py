import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import PIPELINE_MODEL


# Single source of truth: the prompt rule below is generated from this list, and
# eval/checks.py greps drafts for the same strings.
BANNED_PHRASES = [
    "I hope this email finds you well",
    "I came across your profile",
    "I wanted to reach out",
    "touch base",
    "synergy",
]

_BANNED_FOR_PROMPT = ", ".join(f'"{phrase}"' for phrase in BANNED_PHRASES)

SYSTEM_PROMPT = f"""You write hyper-personalised cold emails on behalf of Alchemyst AI —
an AI infrastructure company that helps B2B teams automate sales, marketing, and ops workflows.

Rules (non-negotiable):
1. Max 120 words. Brevity is respect.
2. Open with a specific observation about the company or person — never a generic opener.
3. Never say: {_BANNED_FOR_PROMPT}.
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
        model=PIPELINE_MODEL,
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


REWRITE_TEMPLATE = """Your previous draft did not pass review. Rewrite it.

Lead info:
- Name: {name}
- Role: {role}
- Company: {company}

Research brief (the ONLY source you may draw facts from):
{research}

Your previous draft:
---
{previous_draft}
---

What was wrong with it:
{critique}

Fix every point above. Do not introduce any new claim about the company or person that the
research brief does not state — an unsupported claim is the most serious failure here, and
writing a vaguer email is always better than writing a confident wrong one.
Keep the same rules: subject line first, then the body."""


_rewriter = None


def build_rewriter():
    llm = ChatGroq(
        model=PIPELINE_MODEL,
        # Lower than the drafter's 0.7: a rewrite is a correction against specific feedback,
        # not another creative attempt, and high temperature here tends to reintroduce the
        # invented detail the critique just asked to remove.
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", REWRITE_TEMPLATE),
    ])

    return prompt | llm | StrOutputParser()


def rewrite_email(lead: dict, research: str, previous_draft: str, critique: str) -> str:
    """Second attempt at a draft, given the critic's specific objections."""
    global _rewriter
    if _rewriter is None:
        _rewriter = build_rewriter()

    return _rewriter.invoke({
        "name": lead.get("name", ""),
        "role": lead.get("role", ""),
        "company": lead.get("company", ""),
        "research": research,
        "previous_draft": previous_draft,
        "critique": critique,
    })