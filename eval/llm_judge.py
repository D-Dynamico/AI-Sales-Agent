"""LLM judge for personalisation quality.

Deliberately built on its own client and prompt rather than reusing the drafter's chain,
so a regression in the drafter's prompt can't mark its own output as good.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config import JUDGE_MODEL

SYSTEM_PROMPT = """You grade cold emails on ONE axis: personalisation specificity.

5 — opens on a fact that could only come from the research brief; the email would make no
    sense sent to a different company.
4 — clearly grounded in the brief, but the specifics are mild or could fit close competitors.
3 — references the company or role, but only in ways guessable without any research.
2 — mostly template with a name and company slotted in.
1 — fully generic; the research brief left no trace.

Judge only specificity. Ignore tone, grammar, length, and whether you'd reply to it."""

USER_TEMPLATE = """Lead: {name}, {role} at {company}

Research brief given to the writer:
{research}

Email they produced:
{draft}

Score it."""


class JudgeScore(BaseModel):
    score: int = Field(description="Personalisation specificity, 1 to 5")
    reason: str = Field(description="One sentence naming the specific detail that earned the score")


_judge = None


def _build_judge():
    llm = ChatGroq(model=JUDGE_MODEL, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])
    return prompt | llm.with_structured_output(JudgeScore)


def judge_draft(lead: dict, research: str, draft: str) -> JudgeScore:
    global _judge
    if _judge is None:
        _judge = _build_judge()

    return _judge.invoke({
        "name": lead.get("name", ""),
        "role": lead.get("role", ""),
        "company": lead.get("company", ""),
        "research": research,
        "draft": draft,
    })
