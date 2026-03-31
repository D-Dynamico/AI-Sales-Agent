"""
Email sender — sends via SendGrid. Falls back to SMTP if SENDGRID_API_KEY not set.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


FROM_NAME = "Alex from Alchemyst AI"
FROM_EMAIL = os.getenv("FROM_EMAIL", "alex@alchemyst.ai")


def _parse_subject_body(email_text: str):
    """Split the LLM output into subject and body."""
    lines = email_text.strip().split("\n")
    subject = "Quick question"
    body_lines = []

    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            body_lines = lines[i + 1:]
            break
    else:
        body_lines = lines

    body = "\n".join(body_lines).strip()
    return subject, body


def send_via_sendgrid(to_email: str, to_name: str, subject: str, body: str):
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content

    sg = sendgrid.SendGridAPIClient(api_key=os.getenv("SENDGRID_API_KEY"))
    message = Mail(
        from_email=Email(FROM_EMAIL, FROM_NAME),
        to_emails=To(to_email, to_name),
        subject=subject,
        plain_text_content=Content("text/plain", body),
    )
    response = sg.client.mail.send.post(request_body=message.get())

    if response.status_code not in (200, 202):
        raise RuntimeError(f"SendGrid error: {response.status_code} {response.body}")


def send_via_smtp(to_email: str, to_name: str, subject: str, body: str):
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")

    if not smtp_user or not smtp_pass:
        raise ValueError("SMTP_USER and SMTP_PASS must be set in .env")

    msg = MIMEMultipart()
    msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
    msg["To"] = f"{to_name} <{to_email}>"
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())


def send_email(to_email: str, to_name: str, body: str):
    """
    Main send function. Parses subject from LLM output, then sends.
    Uses SendGrid if SENDGRID_API_KEY is set, otherwise falls back to SMTP.
    """
    subject, clean_body = _parse_subject_body(body)

    if os.getenv("SENDGRID_API_KEY"):
        send_via_sendgrid(to_email, to_name, subject, clean_body)
    else:
        send_via_smtp(to_email, to_name, subject, clean_body)