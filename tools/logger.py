"""
Logger — writes results to a CSV output log and checks for duplicates.
"""

import csv
import os
from datetime import datetime
from pathlib import Path


FIELDNAMES = ["timestamp", "name", "role", "company", "email", "status", "draft_preview"]


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


def log_result(lead: dict, draft: str, status: str, log_path: Path):
    """Appends a result row to the output CSV log."""
    _ensure_log(log_path)

    preview = draft[:200].replace("\n", " ") if draft else ""

    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "name": lead.get("name", ""),
        "role": lead.get("role", ""),
        "company": lead.get("company", ""),
        "email": lead.get("email", ""),
        "status": status,
        "draft_preview": preview,
    }

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
