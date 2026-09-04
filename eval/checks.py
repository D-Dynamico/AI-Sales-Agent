"""Deterministic draft checks. Pure functions, no I/O, free to run."""

from agents.drafter import BANNED_PHRASES

MAX_WORDS = 120


def split_subject_body(draft) -> tuple[str, str]:
    """Accepts an EmailDraft or the legacy "Subject: ...\\n\\nbody" string.

    Structured drafts carry the two fields already, so there is nothing to parse — the
    string branch survives only for logged/stored drafts and the tests that pin the old
    format. New code should pass an EmailDraft and let this be a no-op.
    """
    if hasattr(draft, "subject") and hasattr(draft, "body"):
        return (draft.subject or "").strip(), (draft.body or "").strip()

    lines = str(draft).strip().split("\n")
    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            return line.split(":", 1)[1].strip(), "\n".join(lines[i + 1:]).strip()
    return "", str(draft).strip()


def word_count(draft) -> int:
    """Body only — rule 1's 120-word budget doesn't cover the subject line."""
    _, body = split_subject_body(draft)
    return len(body.split())


def has_subject(draft) -> bool:
    return bool(split_subject_body(draft)[0])


def cliche_hits(draft) -> list[str]:
    subject, body = split_subject_body(draft)
    lowered = f"{subject} {body}".lower()
    return [phrase for phrase in BANNED_PHRASES if phrase.lower() in lowered]


def cta_count(draft) -> int:
    """Heuristic: question marks in the body. Catches the failure we care about —
    a draft stacking several asks — at the cost of counting rhetorical openers."""
    _, body = split_subject_body(draft)
    return body.count("?")


def run_all(draft) -> dict:
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
