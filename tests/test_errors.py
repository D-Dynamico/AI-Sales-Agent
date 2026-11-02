"""Tests for user-facing error messages and redaction.

The message a viewer of the public Streamlit demo sees is the whole point here. Groq's raw
rate-limit error carries the organisation id, the exact quota and a billing upsell — the org
id especially has no business being rendered on a page anyone can open.
"""

import pytest

from agents.errors import classify, log_detail, redact, retry_hint, user_message

# Verbatim shape of the real error, minus the actual org id.
GROQ_TPD = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_01kk7bfaxgf0b8x34yhfzht3j6` service tier "
    "`on_demand` on tokens per day (TPD): Limit 200000, Used 199528, Requested 2989. "
    "Please try again in 18m7.344s. Need more tokens? Upgrade to Dev Tier today at "
    "https://console.groq.com/settings/billing', 'type': 'tokens'}}"
)


class TestClassification:
    @pytest.mark.parametrize("text,expected", [
        (GROQ_TPD, "capacity"),
        ("Error code: 429 - rate limit reached", "capacity"),
        ("Error code: 401 - authentication failed", "auth"),
        ("Error code: 503 - service unavailable", "upstream"),
        ("connection timed out", "upstream"),
        ("something entirely unexpected", "unknown"),
    ])
    def test_categories(self, text, expected):
        assert classify(RuntimeError(text)) == expected


class TestUserMessage:
    def test_capacity_message_is_calm_and_specific(self):
        message = user_message(RuntimeError(GROQ_TPD))
        assert message == "Model capacity for today is used up. Try again in about 19 min."

    def test_capacity_message_leaks_nothing(self):
        """The failure this exists to prevent: account details on a public page."""
        message = user_message(RuntimeError(GROQ_TPD))
        for leak in ["org_01kk7bfaxgf0b8x34yhfzht3j6", "console.groq.com", "Dev Tier",
                     "200000", "Upgrade", "429"]:
            assert leak not in message, f"{leak!r} leaked into a user-facing message"

    def test_falls_back_gracefully_without_a_retry_hint(self):
        assert "Try again shortly." in user_message(RuntimeError("429 rate limit reached"))

    def test_step_is_named_where_it_helps(self):
        assert "researching this lead" in user_message(RuntimeError("boom"),
                                                       "researching this lead")

    def test_auth_and_upstream_read_plainly(self):
        assert "credentials" in user_message(RuntimeError("401 authentication"))
        assert "temporarily unavailable" in user_message(RuntimeError("503 overloaded"))

    def test_no_message_ends_up_empty(self):
        assert user_message(RuntimeError("")).strip()


class TestRedaction:
    def test_strips_org_id_billing_link_and_upsell(self):
        cleaned = redact(GROQ_TPD)
        assert "org_01kk7bfaxgf0b8x34yhfzht3j6" not in cleaned
        assert "console.groq.com" not in cleaned
        assert "Upgrade to Dev Tier" not in cleaned

    def test_no_url_fragment_survives(self):
        """Regression: removing the upsell first cut the URL at the dot in its hostname
        and left ".groq.com/settings/billing" sitting in the log."""
        cleaned = redact(GROQ_TPD)
        assert "groq.com" not in cleaned
        assert "settings/billing" not in cleaned

    def test_keeps_what_makes_it_debuggable(self):
        """Redaction must not blind the operator — the cause has to survive."""
        cleaned = redact(GROQ_TPD)
        assert "429" in cleaned
        assert "tokens per day" in cleaned.lower()

    @pytest.mark.parametrize("secret", [
        "gsk_abcdefghijklmnop", "tvly-dev-abcdefghijk", "SG.abcdefghijklmno",
    ])
    def test_api_keys_never_survive_redaction(self, secret):
        assert secret not in redact(f"auth failed with key {secret} sent")

    def test_empty_input_is_handled(self):
        assert redact("") == "unspecified error"


class TestLogDetail:
    def test_includes_type_and_stays_short(self):
        detail = log_detail(RuntimeError(GROQ_TPD))
        assert detail.startswith("RuntimeError:")
        assert len(detail) <= 300
        assert "org_01kk7bfaxgf0b8x34yhfzht3j6" not in detail

    def test_is_single_line_for_csv_safety(self):
        assert "\n" not in log_detail(RuntimeError("line one\nline two"))


def test_retry_hint_extraction():
    assert retry_hint(RuntimeError(GROQ_TPD)) == "18m7.344s"
    assert retry_hint(RuntimeError("no hint here")) == ""


def test_wait_is_rounded_up_to_readable_minutes():
    from agents.errors import humanize_wait
    assert humanize_wait("18m7.344s") == "19 min"
    assert humanize_wait("45.2s") == "a minute"
    assert humanize_wait("1h2m3s") == "1 h 3 min"
    assert humanize_wait("2h0m0s") == "2 h"
    assert humanize_wait("") == ""
