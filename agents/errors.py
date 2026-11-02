"""Turns provider exceptions into something safe to show a person.

Raw Groq errors are not presentable. A rate-limit message carries the organisation id, the
exact quota, and a "Upgrade to Dev Tier today" billing link — none of which a viewer of the
public demo should see, and the org id in particular is not something to print on a page
anyone can open.

Two separate jobs here, deliberately kept apart:
  - `user_message()` produces a short, calm sentence for humans.
  - `redact()` strips identifiers from technical text that still gets logged, so the detail
    needed for debugging survives without the account details riding along.
"""

import re

CAPACITY_MARKERS = ("429", "rate limit", "rate_limit", "tokens per day", "(tpd)",
                    "quota", "insufficient_quota")
AUTH_MARKERS = ("401", "403", "authentication", "invalid api key", "unauthorized")
UPSTREAM_MARKERS = ("500", "502", "503", "504", "overloaded", "timeout", "timed out")

_ORG_ID = re.compile(r"org_[A-Za-z0-9]+")
_BILLING_URL = re.compile(r"https?://\S*(?:billing|console|upgrade)\S*", re.IGNORECASE)
_UPGRADE_PITCH = re.compile(r"need more tokens\?.*?(?=['\"]|$)", re.IGNORECASE | re.DOTALL)
# The separator varies by vendor — Groq uses "gsk_", Tavily "tvly-", SendGrid "SG." — so
# all three are accepted. Missing the dot meant a live SendGrid key would have survived.
_API_KEY = re.compile(r"\b(?:gsk|sk|tvly|SG)[-_.][A-Za-z0-9._-]{8,}")


def classify(exc: Exception) -> str:
    """capacity | auth | upstream | unknown."""
    text = str(exc).lower()
    if any(marker in text for marker in CAPACITY_MARKERS):
        return "capacity"
    if any(marker in text for marker in AUTH_MARKERS):
        return "auth"
    if any(marker in text for marker in UPSTREAM_MARKERS):
        return "upstream"
    return "unknown"


def redact(text: str) -> str:
    """Remove account identifiers, credentials and vendor upsell from technical text."""
    cleaned = _API_KEY.sub("[redacted]", str(text))
    cleaned = _ORG_ID.sub("[org]", cleaned)
    # URL first: the upgrade-pitch pattern would otherwise stop at the dot inside the
    # hostname and leave half a link ("...groq.com/settings/billing") behind.
    cleaned = _BILLING_URL.sub("", cleaned)
    cleaned = _UPGRADE_PITCH.sub("", cleaned)
    return " ".join(cleaned.split()).strip(" .,") or "unspecified error"


def retry_hint(exc: Exception) -> str:
    """Pull the provider's own "try again in 3m56s" out, since it's genuinely useful."""
    match = re.search(r"try again in ([\dhms.]+)", str(exc), re.IGNORECASE)
    # The character class swallows the sentence's full stop; "18m7.344s." reads as a typo.
    return match.group(1).rstrip(".") if match else ""


def humanize_wait(hint: str) -> str:
    """'18m7.344s' -> '19 min', '1h2m' -> '1 h 2 min', '45.2s' -> 'a minute'.

    Rounded up: telling someone to retry a few seconds early just earns them another 429."""
    import math
    parts = {unit: value for value, unit in re.findall(r"([\d.]+)([hms])", hint or "")}
    if not parts:
        return ""
    seconds = (float(parts.get("h", 0)) * 3600 + float(parts.get("m", 0)) * 60
               + float(parts.get("s", 0)))
    minutes = max(1, math.ceil(seconds / 60))
    if minutes == 1:
        return "a minute"
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours} h {mins} min" if mins else f"{hours} h"
    return f"{mins} min"


def user_message(exc: Exception, step: str = "") -> str:
    """A short sentence safe to show anyone, including on a public page."""
    where = f" while {step}" if step else ""
    kind = classify(exc)

    if kind == "capacity":
        wait = humanize_wait(retry_hint(exc))
        when = f" Try again in about {wait}." if wait else " Try again shortly."
        return f"Model capacity for today is used up.{when}"
    if kind == "auth":
        return f"The model provider rejected our credentials{where}."
    if kind == "upstream":
        return f"The model provider is temporarily unavailable{where}. Try again shortly."
    return f"Something went wrong{where}."


def log_detail(exc: Exception) -> str:
    """Compact, redacted technical detail for the CSV log and the terminal."""
    return f"{type(exc).__name__}: {redact(exc)}"[:300]
