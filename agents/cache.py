"""Per-company research cache.

Research is per *company*, but the pipeline runs per *lead*, so ten contacts at one firm
previously cost ten identical research passes. That was wasteful in the abstract before;
it's now the binding constraint, since the models sit behind a 200k tokens-per-day cap and
the Phase 1 critic roughly doubled the spend per lead.

Deliberately a plain JSON-file cache rather than a dependency: the volumes here are tiny,
and Phase 4's SQLite migration will absorb this into a real table. Reaching for `diskcache`
now would mean adding a dependency with a three-week shelf life.
"""

import json
import re
import time
from pathlib import Path

CACHE_DIR = Path("data/research_cache")

# A week: long enough that a batch of leads at one company costs one pass, short enough
# that "recent news" stays recent. Tavily is already filtering to the last 30 days, so a
# stale-but-cached result can only ever be a few days behind the live query.
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60


def _slug(company: str) -> str:
    """Normalise so "Razorpay", "razorpay " and "RAZORPAY" share one entry."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", (company or "").strip().lower()).strip("-")
    return cleaned or "unknown"


def _path(company: str, cache_dir: Path | None = None) -> Path:
    return (cache_dir or CACHE_DIR) / f"{_slug(company)}.json"


def get(company: str, ttl_seconds: int = DEFAULT_TTL_SECONDS,
        cache_dir: Path | None = None) -> dict | None:
    """Return the cached payload for a company, or None if absent, stale or unreadable.

    A corrupt cache file must never break a run — it's an optimisation, so any failure to
    read is treated as a miss.
    """
    path = _path(company, cache_dir)
    if not path.exists():
        return None

    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - entry["cached_at"] > ttl_seconds:
            return None
        return entry["payload"]
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return None


def set(company: str, payload: dict, cache_dir: Path | None = None) -> None:
    """Cache a company's research. Failures are swallowed for the same reason as above."""
    path = _path(company, cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"cached_at": time.time(), "company": company, "payload": payload},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def clear(cache_dir: Path | None = None) -> int:
    """Drop every cached entry. Returns how many were removed."""
    directory = cache_dir or CACHE_DIR
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.glob("*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
