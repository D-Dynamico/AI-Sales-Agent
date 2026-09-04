"""Tests for send routing and subject parsing. No network calls are made."""

import pytest

from tools import sender
from tools.sender import _parse_subject_body


class TestSubjectParsing:
    def test_splits_subject_and_body(self):
        subject, body = _parse_subject_body("Subject: Quick one\n\nHello there.\n\nAlex")
        assert subject == "Quick one"
        assert body.startswith("Hello there.")
        assert "Subject:" not in body

    def test_falls_back_when_no_subject_line(self):
        subject, body = _parse_subject_body("Hello there, no subject.")
        assert subject == "Quick question"
        assert body == "Hello there, no subject."

    def test_handles_colons_inside_the_subject(self):
        subject, _ = _parse_subject_body("Subject: Re: your launch\n\nBody")
        assert subject == "Re: your launch"


class TestSendRouting:
    def test_uses_sendgrid_when_key_present(self, monkeypatch):
        monkeypatch.setenv("SENDGRID_API_KEY", "SG.fake")
        called = {}
        monkeypatch.setattr(sender, "send_via_sendgrid",
                            lambda *a, **k: called.setdefault("sendgrid", (a, k)))
        monkeypatch.setattr(sender, "send_via_smtp",
                            lambda *a, **k: pytest.fail("SMTP must not be used"))

        sender.send_email("a@b.com", "A", "Subject: Hi\n\nBody")

        assert "sendgrid" in called

    def test_falls_back_to_smtp_without_key(self, monkeypatch):
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        called = {}
        monkeypatch.setattr(sender, "send_via_smtp",
                            lambda *a, **k: called.setdefault("smtp", (a, k)))
        monkeypatch.setattr(sender, "send_via_sendgrid",
                            lambda *a, **k: pytest.fail("SendGrid must not be used"))

        sender.send_email("a@b.com", "A", "Subject: Hi\n\nBody")

        assert "smtp" in called

    def test_parsed_subject_reaches_the_transport(self, monkeypatch):
        """The transport must receive a clean subject and a body with the header stripped."""
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        captured = {}
        monkeypatch.setattr(sender, "send_via_smtp",
                            lambda to, name, subject, body: captured.update(
                                subject=subject, body=body))

        sender.send_email("a@b.com", "A", "Subject: Launch congrats\n\nReal body here.")

        assert captured["subject"] == "Launch congrats"
        assert "Subject:" not in captured["body"]
        assert captured["body"] == "Real body here."


class TestSmtpGuards:
    def test_missing_credentials_raise_before_connecting(self, monkeypatch):
        monkeypatch.delenv("SMTP_USER", raising=False)
        monkeypatch.delenv("SMTP_PASS", raising=False)
        monkeypatch.setattr(sender.smtplib, "SMTP",
                            lambda *a, **k: pytest.fail("must not connect without credentials"))

        with pytest.raises(ValueError, match="SMTP_USER"):
            sender.send_via_smtp("a@b.com", "A", "Subj", "Body")
