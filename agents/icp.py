"""Ideal-customer-profile scoring, used to skip poor-fit leads before spending tokens.

Deliberately rule-based and LLM-free. An LLM scorer would burn tokens on exactly the leads
we're trying to avoid spending tokens on, which defeats the purpose — the filter has to be
cheaper than the work it prevents. Rules also make the decision inspectable: a skipped lead
comes with the reason it was skipped, which matters when the thing being automated is
"don't contact this person".

The keyword lists are the ICP definition for now; Phase 6 moves them into persona config
alongside the value proposition, since "who we sell to" belongs with "what we sell".
"""

from dataclasses import dataclass, field

# Founders and C-level own every function, so "is this their remit?" is always yes for them.
# Scoring them on function keywords was wrong: it dropped a co-founder to a partial match
# because "founder" isn't a department, which is exactly backwards for who signs a deal.
EXEC_TITLES = {
    "founder", "co-founder", "cofounder", "ceo", "cto", "coo", "cfo", "cmo", "cro", "cpo",
    "chief", "president", "owner", "partner",
}

# Roles with budget or direct ownership of the problem this product addresses.
SENIOR_TITLES = {
    "chief", "cxo", "ceo", "cto", "coo", "cfo", "cmo", "cro", "cpo",
    "founder", "co-founder", "cofounder", "president", "partner", "owner",
    "vp", "vice", "head", "director", "principal", "lead", "chief of staff",
}

# Functions this product is actually for: sales, marketing, revenue, growth, ops.
RELEVANT_FUNCTIONS = {
    "sales", "marketing", "revenue", "growth", "demand", "outbound", "gtm",
    "operations", "ops", "business", "bizdev", "partnerships", "commercial",
    "customer", "success", "strategy",
}

# Roles that are individual contributors, too junior to buy, or plainly out of scope.
DISQUALIFYING_TITLES = {
    "intern", "trainee", "student", "apprentice", "assistant", "associate",
    "recruiter", "volunteer", "retired", "freelance",
}


@dataclass
class ICPScore:
    score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons) or "no signals"


def _tokens(text: str) -> set[str]:
    return {t.strip(".,()-/") for t in (text or "").lower().replace("/", " ").split()}


def score_icp(lead: dict) -> ICPScore:
    """Score a lead 0-1 on fit. Cheap, deterministic, and explainable."""
    role = (lead.get("role") or "").strip()
    company = (lead.get("company") or "").strip()
    email = (lead.get("email") or "").strip()
    tokens = _tokens(role)

    score = 0.0
    reasons = []

    if not email:
        return ICPScore(0.0, ["no email address"])
    if not company:
        return ICPScore(0.0, ["no company"])

    disqualifying = tokens & DISQUALIFYING_TITLES
    if disqualifying:
        return ICPScore(0.0, [f"disqualifying title: {', '.join(sorted(disqualifying))}"])

    if not role:
        # No title is weak signal rather than a disqualification — plenty of exported lead
        # lists lose the field. Let it through at a low score for a human to weigh.
        return ICPScore(0.3, ["no role given"])

    executive = tokens & EXEC_TITLES
    if executive:
        return ICPScore(1.0, [f"executive ({', '.join(sorted(executive))}) — owns every function"])

    seniority = tokens & SENIOR_TITLES
    if seniority:
        score += 0.5
        reasons.append(f"senior title ({', '.join(sorted(seniority))})")
    else:
        reasons.append("no seniority signal in title")

    function = tokens & RELEVANT_FUNCTIONS
    if function:
        score += 0.5
        reasons.append(f"relevant function ({', '.join(sorted(function))})")
    else:
        reasons.append("function not in target set")

    return ICPScore(round(min(score, 1.0), 2), reasons)


def passes_icp(lead: dict, threshold: float) -> tuple[bool, ICPScore]:
    result = score_icp(lead)
    return result.score >= threshold, result
