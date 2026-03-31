import argparse
import csv
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from agents.enricher import enrich_lead
from agents.drafter import draft_email
from tools.sender import send_email
from tools.logger import log_result, already_sent

load_dotenv()

LEADS_FILE = Path("data/leads.csv")
OUTPUT_LOG = Path("data/output_log.csv")


def load_leads():
    with open(LEADS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_pipeline(dry_run=False, resume=False, delay=60):
    leads = load_leads()
    print(f"\nStarting pipeline for {len(leads)} leads | dry_run={dry_run} | resume={resume}\n")

    for i, lead in enumerate(leads):
        email = lead.get("email", "").strip()
        name = lead.get("name", "Unknown")

        if not email:
            print(f"[{i+1}] Skipping {name} - no email address")
            continue

        if resume and already_sent(email, OUTPUT_LOG):
            print(f"[{i+1}] Skipping {name} - already sent")
            continue

        print(f"[{i+1}] Enriching lead: {name} @ {lead.get('company', '')}")
        try:
            research = enrich_lead(lead)
            print(f"     Research: {research[:120]}...")
        except Exception as e:
            print(f"     Enrichment failed: {e}")
            log_result(lead, draft="", status=f"ENRICH_ERROR: {e}", log_path=OUTPUT_LOG)
            continue

        print(f"[{i+1}] Drafting email...")
        try:
            email_body = draft_email(lead, research)
            print(f"     Preview: {email_body[:100]}...")
        except Exception as e:
            print(f"     Draft failed: {e}")
            log_result(lead, draft="", status=f"DRAFT_ERROR: {e}", log_path=OUTPUT_LOG)
            continue

        if dry_run:
            print(f"[{i+1}] [DRY RUN] Would send to {email}:\n")
            print("-" * 60)
            print(email_body)
            print("-" * 60 + "\n")
            log_result(lead, draft=email_body, status="DRY_RUN", log_path=OUTPUT_LOG)
        else:
            print(f"[{i+1}] Sending to {email}...")
            try:
                send_email(to_email=email, to_name=name, body=email_body)
                print("     Sent successfully")
                log_result(lead, draft=email_body, status="SENT", log_path=OUTPUT_LOG)
            except Exception as e:
                print(f"     Send failed: {e}")
                log_result(lead, draft=email_body, status=f"SEND_ERROR: {e}", log_path=OUTPUT_LOG)

        if i < len(leads) - 1 and not dry_run:
            print(f"     Waiting {delay} seconds before next send...\n")
            time.sleep(delay)

    print("\nPipeline complete. Results logged to", OUTPUT_LOG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Sales Outreach Agent")
    parser.add_argument("--dry-run", action="store_true", help="Print emails instead of sending")
    parser.add_argument("--resume", action="store_true", help="Skip leads already in output log")
    parser.add_argument("--delay", type=int, default=60, help="Seconds between sends (default: 60)")
    args = parser.parse_args()

    run_pipeline(dry_run=args.dry_run, resume=args.resume, delay=args.delay)