"""Deterministic draft checks. Pure functions, no I/O, free to run."""

from agents.drafter import BANNED_PHRASES

MAX_WORDS = 120


def split_subject_body(draft: str) -> tuple[str, str]:
    lines = draft.strip().split("\n")
    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            return line.split(":", 1)[1].strip(), "\n".join(lines[i + 1:]).strip()
    return "", draft.strip()


def word_count(draft: str) -> int:
    """Body only — rule 1's 120-word budget doesn't cover the subject line."""
    _, body = split_subject_body(draft)
    return len(body.split())


def has_subject(draft: str) -> bool:
    return bool(split_subject_body(draft)[0])


def cliche_hits(draft: str) -> list[str]:
    lowered = draft.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase.lower() in lowered]


def cta_count(draft: str) -> int:
    """Heuristic: question marks in the body. Catches the failure we care about —
    a draft stacking several asks — at the cost of counting rhetorical openers."""
    _, body = split_subject_body(draft)
    return body.count("?")


def run_all(draft: str) -> dict:
    hits = cliche_hits(draft)
    words = word_count(draft)
    ctas = cta_count(draft)
    return {
        "word_count": words,
        "word_count_pass": words <= MAX_WORDS,
        "subject_pass": has_subject(draft),
        "cliche_hits": hits,
        "cliche_pass": not hits,
        "cta_count": ctas,
        "cta_pass": ctas == 1,
    }
