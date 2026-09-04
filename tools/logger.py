"""
Logger — writes results to a CSV output log and checks for duplicates.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path


FIELDNAMES = [
    "timestamp", "name", "role", "company", "email", "status", "draft_preview", "critique",
]


def _ensure_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def already_sent(email: str, log_path: Path) -> bool:
    """Returns True if this email address already has a SENT entry in the log."""
    if not log_path.exists():
        return False
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("email") == email and row.get("status") == "SENT":
                return True
    return False


def log_result(lead: dict, draft: str, status: str, log_path: Path, critique: str = ""):
    """Appends a result row to the output CSV log.

    `critique` carries the critic's objections for NEEDS_REVIEW rows, so the review queue
    can be worked straight from this file without re-running the pipeline.
    """
    _ensure_log(log_path)

    rendered = draft.render() if hasattr(draft, "render") else (draft or "")
    preview = rendered[:200].replace("\n", " ")

    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "name": lead.get("name", ""),
        "role": lead.get("role", ""),
        "company": lead.get("company", ""),
        "email": lead.get("email", ""),
        "status": status,
        "draft_preview": preview,
        "critique": critique.replace("\n", " | ") if critique else "",
    }

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
