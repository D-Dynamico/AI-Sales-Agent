"""Turns the enricher's raw source strings back into something presentable.

The enricher stores each search result as one string — "title · date", the page text, then
"[url]" — because that is what the drafter and critic read. Shown as-is, that string is
scraped page furniture: markdown image tags, "Image 10:" captions, nav links and headings.
This splits it into its parts and cleans the snippet for display only; the stored text the
critic checks claims against is left untouched.
"""

import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_IMAGE_CAPTION = re.compile(r"\bImage \d+:\s*", re.IGNORECASE)
_BARE_URL = re.compile(r"https?://\S+")
_MD_MARKS = re.compile(r"(^|\s)[#>*_|]+(?=\s)|[*_]{2,}")


@dataclass(frozen=True)
class Source:
    title: str
    date: str
    url: str
    domain: str
    snippet: str


def _format_date(raw: str) -> str:
    """'Sun, 30 Aug 2026 07:00:54 GMT' -> '30 Aug 2026'. Anything unparseable is kept."""
    try:
        dt = parsedate_to_datetime(raw)
        return f"{dt.day} {dt:%b %Y}"
    except Exception:
        return raw.strip()


def clean_snippet(text: str, limit: int = 240) -> str:
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _IMAGE_CAPTION.sub("", text)
    text = _BARE_URL.sub("", text)
    text = _MD_MARKS.sub(" ", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return cut + "…"


def parse_source(raw: str) -> Source:
    lines = [ln for ln in (raw or "").strip().splitlines()]
    url = ""
    if lines and lines[-1].startswith("[") and lines[-1].endswith("]"):
        url = lines.pop()[1:-1].strip()

    title, date = "", ""
    if len(lines) > 1:
        header = lines.pop(0)
        title, _, date = header.partition(" · ")
        title, date = title.strip(), _format_date(date) if date else ""

    domain = urlparse(url).netloc.removeprefix("www.") if url else ""
    return Source(title=title or domain or "Untitled result", date=date, url=url,
                  domain=domain, snippet=clean_snippet("\n".join(lines)))
