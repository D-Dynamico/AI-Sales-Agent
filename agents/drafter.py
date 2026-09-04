from agents.schemas import EmailDraft
from persona import PERSONA
from agents.structured import StructuredChain


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

SYSTEM_PROMPT = f"""You write hyper-personalised cold emails on behalf of {PERSONA.company} —
{PERSONA.offering}.

Rules (non-negotiable):
1. Max 120 words. Brevity is respect.
2. Open with a specific observation about the company or person — never a generic opener.
3. Never say: {_BANNED_FOR_PROMPT}.
4. One clear, low-friction CTA (e.g. "Worth a 15-min call this week?")
5. No bullet points. Flowing prose only.
6. Sign off as: {PERSONA.signoff}
7. Subject line: punchy, specific, under 8 words. Return it in the `subject` field.
8. The `body` field contains the email only. Never repeat the subject inside the body.

Tone: Direct. Intelligent. Peer-to-peer. Not salesy.
"""

USER_TEMPLATE = """Write a cold email for this lead using the research below.

Lead info:
- Name: {name}
- Role: {role}
- Company: {company}

Research brief:
{research}

"""


_JSON_HINT = ("Reply with ONLY a JSON object of the form "
              '{{"subject": "...", "body": "..."}} — no prose, no code fences.')

_drafter = None


def build_drafter():
    return StructuredChain(
        EmailDraft, SYSTEM_PROMPT, USER_TEMPLATE,
        temperature=0.7, json_hint=_JSON_HINT,
        # A silently empty email is worse than a logged failure: the pipeline would send
        # blank mail. Let it surface as DRAFT_ERROR instead.
        on_unparseable="raise",
    )


def draft_email(lead: dict, research: str) -> EmailDraft:
    """Draft an email as structured subject + body."""
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
Keep the same rules, and keep the subject and body in their separate fields."""


_rewriter = None


def build_rewriter():
    return StructuredChain(
        EmailDraft, SYSTEM_PROMPT, REWRITE_TEMPLATE,
        # Lower than the drafter's 0.7: a rewrite is a correction against specific feedback,
        # not another creative attempt, and high temperature here tends to reintroduce the
        # invented detail the critique just asked to remove.
        temperature=0.3, json_hint=_JSON_HINT, on_unparseable="raise",
    )


def rewrite_email(lead: dict, research: str, previous_draft, critique: str) -> EmailDraft:
    """Second attempt at a draft, given the critic's specific objections."""
    global _rewriter
    if _rewriter is None:
        _rewriter = build_rewriter()

    previous = previous_draft.render() if hasattr(previous_draft, "render") else previous_draft
    return _rewriter.invoke({
        "name": lead.get("name", ""),
        "role": lead.get("role", ""),
        "company": lead.get("company", ""),
        "research": research,
        "previous_draft": previous,
        "critique": critique,
    })
