"""Tests for the per-company research cache.

The cache exists to protect a hard daily token budget, so the behaviour that matters is
that a hit really avoids the work, and that a broken cache degrades to a miss instead of
taking a run down with it.
"""

import json
import time

from agents import cache


def test_round_trips_a_payload(tmp_path):
    cache.set("Razorpay", {"company": "Razorpay", "confidence": 0.8}, cache_dir=tmp_path)
    assert cache.get("Razorpay", cache_dir=tmp_path)["confidence"] == 0.8


def test_miss_returns_none(tmp_path):
    assert cache.get("Nobody", cache_dir=tmp_path) is None


def test_company_name_is_normalised(tmp_path):
    """Ten leads at one firm must share one entry even when the CSV spells it unevenly."""
    cache.set("Razorpay", {"v": 1}, cache_dir=tmp_path)

    for variant in ["razorpay", "RAZORPAY", "  Razorpay  ", "Razorpay!"]:
        assert cache.get(variant, cache_dir=tmp_path) == {"v": 1}


def test_distinct_companies_do_not_collide(tmp_path):
    cache.set("Zepto", {"v": "zepto"}, cache_dir=tmp_path)
    cache.set("Meesho", {"v": "meesho"}, cache_dir=tmp_path)

    assert cache.get("Zepto", cache_dir=tmp_path)["v"] == "zepto"
    assert cache.get("Meesho", cache_dir=tmp_path)["v"] == "meesho"


def test_expired_entry_is_a_miss(tmp_path):
    cache.set("Razorpay", {"v": 1}, cache_dir=tmp_path)
    assert cache.get("Razorpay", ttl_seconds=0, cache_dir=tmp_path) is None


def test_fresh_entry_within_ttl_is_a_hit(tmp_path):
    cache.set("Razorpay", {"v": 1}, cache_dir=tmp_path)
    assert cache.get("Razorpay", ttl_seconds=3600, cache_dir=tmp_path) is not None


def test_stored_timestamp_is_used_for_expiry(tmp_path):
    path = tmp_path / "razorpay.json"
    path.write_text(json.dumps({"cached_at": time.time() - 5000, "payload": {"v": 1}}),
                    encoding="utf-8")

    assert cache.get("Razorpay", ttl_seconds=10000, cache_dir=tmp_path) == {"v": 1}
    assert cache.get("Razorpay", ttl_seconds=1000, cache_dir=tmp_path) is None


class TestCorruptCacheDegradesToMiss:
    """A cache is an optimisation; no failure to read it may break a run."""

    def test_malformed_json(self, tmp_path):
        (tmp_path / "razorpay.json").write_text("{not json", encoding="utf-8")
        assert cache.get("Razorpay", cache_dir=tmp_path) is None

    def test_missing_expected_keys(self, tmp_path):
        (tmp_path / "razorpay.json").write_text(json.dumps({"unexpected": True}),
                                                encoding="utf-8")
        assert cache.get("Razorpay", cache_dir=tmp_path) is None

    def test_non_numeric_timestamp(self, tmp_path):
        (tmp_path / "razorpay.json").write_text(
            json.dumps({"cached_at": "yesterday", "payload": {}}), encoding="utf-8")
        assert cache.get("Razorpay", cache_dir=tmp_path) is None


def test_empty_company_name_is_handled(tmp_path):
    cache.set("", {"v": 1}, cache_dir=tmp_path)
    assert cache.get("", cache_dir=tmp_path) == {"v": 1}


def test_clear_removes_entries(tmp_path):
    cache.set("A", {"v": 1}, cache_dir=tmp_path)
    cache.set("B", {"v": 2}, cache_dir=tmp_path)

    assert cache.clear(cache_dir=tmp_path) == 2
    assert cache.get("A", cache_dir=tmp_path) is None


def test_clear_on_missing_directory(tmp_path):
    assert cache.clear(cache_dir=tmp_path / "nope") == 0
