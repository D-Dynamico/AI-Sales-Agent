"""Tests for the assembled enrichment step: plan → parallel search → structure → cache.

Covers the seams between the new pieces, which is where the behaviour that costs money
lives — a cache that doesn't prevent a call, or an unquoted event reaching the drafter.
"""

import pytest

from agents import enricher
from agents.schemas import ResearchResult, TriggerEvent


class FakeSearch:
    def __init__(self, payload=None):
        self.calls = 0
        self.payload = payload or {"results": [
            {"content": "Razorpay launched RAY with IndusInd Bank.",
             "title": "Razorpay launches RAY", "url": "u", "published_date": "03 Sep 2026"},
        ]}

    def invoke(self, query):
        self.calls += 1
        return self.payload


def _patch_search(monkeypatch, search):
    monkeypatch.setattr(enricher, "TavilySearch", lambda **kw: search)


def _patch_structurer(monkeypatch, result, counter=None):
    class FakeStructurer:
        def invoke(self, payload):
            if counter is not None:
                counter["calls"] += 1
            return result

    monkeypatch.setattr(enricher, "_structurer", FakeStructurer())


class TestCaching:
    def test_second_lead_at_same_company_skips_all_work(self, monkeypatch, tmp_path, lead):
        """The point of the cache: ten contacts at one firm cost one research pass."""
        search = FakeSearch()
        counter = {"calls": 0}
        _patch_search(monkeypatch, search)
        _patch_structurer(monkeypatch, ResearchResult(
            company="Razorpay",
            trigger_events=[TriggerEvent(kind="product_launch", summary="RAY launch",
                                         source_quote="Razorpay launched RAY")],
            confidence=0.8), counter)
        monkeypatch.setattr(enricher.cache, "CACHE_DIR", tmp_path)

        first = enricher.research_company(lead)
        second = enricher.research_company({**lead, "name": "Someone Else"})

        assert search.calls > 0
        assert counter["calls"] == 1, "structuring LLM should run once, not per lead"
        assert second.trigger_events[0].summary == first.trigger_events[0].summary

    def test_cache_can_be_disabled(self, monkeypatch, tmp_path, lead):
        """Evaluating an enrichment change needs fresh lookups, or the cache would serve
        results produced by the previous prompt version."""
        counter = {"calls": 0}
        _patch_search(monkeypatch, FakeSearch())
        _patch_structurer(monkeypatch, ResearchResult(company="Razorpay"), counter)
        monkeypatch.setattr(enricher.cache, "CACHE_DIR", tmp_path)

        enricher.research_company(lead, use_cache=False)
        enricher.research_company(lead, use_cache=False)

        assert counter["calls"] == 2


class TestGroundingAtSource:
    def test_events_without_a_source_quote_are_dropped(self, monkeypatch, tmp_path, lead):
        """An unquotable event is one the model invented; it must not reach the drafter,
        where it would become a claim the critic then has to reject."""
        _patch_search(monkeypatch, FakeSearch())
        _patch_structurer(monkeypatch, ResearchResult(company="Razorpay", trigger_events=[
            TriggerEvent(kind="funding", summary="Raised $400M", source_quote=""),
            TriggerEvent(kind="product_launch", summary="Launched RAY",
                         source_quote="Razorpay launched RAY"),
        ]))
        monkeypatch.setattr(enricher.cache, "CACHE_DIR", tmp_path)

        result = enricher.research_company(lead, use_cache=False)

        assert len(result.trigger_events) == 1
        assert result.trigger_events[0].summary == "Launched RAY"

    def test_raw_sources_are_attached_for_the_critic(self, monkeypatch, tmp_path, lead):
        _patch_search(monkeypatch, FakeSearch())
        _patch_structurer(monkeypatch, ResearchResult(company="Razorpay"))
        monkeypatch.setattr(enricher.cache, "CACHE_DIR", tmp_path)

        result = enricher.research_company(lead, use_cache=False)

        assert result.raw_sources
        assert "RAY" in result.raw_sources[0]


class TestNoResearchPath:
    def test_empty_search_never_calls_the_model(self, monkeypatch, tmp_path, lead):
        _patch_search(monkeypatch, FakeSearch(payload={"results": []}))
        monkeypatch.setattr(enricher.cache, "CACHE_DIR", tmp_path)

        def explode(payload):
            raise AssertionError("model must not be asked to summarise nothing")

        monkeypatch.setattr(enricher, "_structurer", type("S", (), {"invoke": staticmethod(explode)}))

        result = enricher.research_company(lead, use_cache=False)

        assert result.trigger_events == []
        assert result.raw_sources == []

    def test_brief_says_so_plainly(self):
        assert "No recent search results" in ResearchResult(company="A").to_brief()


class TestStructuringDelegation:
    """The fallback/error-classification logic itself is tested in test_structured.py.
    What matters here is that the enricher delegates to it and doesn't re-wrap failures."""

    def test_rate_limit_during_structuring_reaches_the_caller(self, monkeypatch, tmp_path, lead):
        """An enrichment 429 must surface as ENRICH_ERROR, not as "no research found" —
        otherwise the pipeline reports an empty web for a question it never asked."""
        _patch_search(monkeypatch, FakeSearch())
        monkeypatch.setattr(enricher, "_structurer", type("S", (), {
            "invoke": lambda self, p: (_ for _ in ()).throw(
                RuntimeError("Error code: 429 - tokens per day (TPD)"))})())

        with pytest.raises(RuntimeError, match="429"):
            enricher.research_company(lead, use_cache=False)


class TestDeduplication:
    def test_same_article_from_two_angles_is_counted_once(self, monkeypatch, tmp_path, lead):
        """Angles overlap; paying to send the same article twice wastes the token budget."""
        duplicate = {"results": [
            {"content": "Razorpay launched RAY.", "title": "T", "url": "u"},
            {"content": "Razorpay launched RAY.", "title": "T", "url": "u"},
        ]}
        _patch_search(monkeypatch, FakeSearch(payload=duplicate))
        _patch_structurer(monkeypatch, ResearchResult(company="Razorpay"))
        monkeypatch.setattr(enricher.cache, "CACHE_DIR", tmp_path)

        result = enricher.research_company(lead, use_cache=False)

        assert len(result.raw_sources) == 1


class TestBriefRendering:
    def test_pain_hypotheses_are_labelled_as_inference(self):
        """Unlabelled, the drafter restates them as fact and the critic rejects the draft."""
        brief = ResearchResult(
            company="A",
            trigger_events=[TriggerEvent(summary="Launched X", source_quote="q")],
            pain_hypotheses=["scaling outbound is hard"],
        ).to_brief()

        assert "Launched X" in brief
        assert "inference" in brief.lower()

    def test_enrich_lead_still_returns_brief_and_sources(self, monkeypatch, tmp_path, lead):
        """The wrapper keeps the Phase 1 contract so callers didn't all have to change."""
        _patch_search(monkeypatch, FakeSearch())
        _patch_structurer(monkeypatch, ResearchResult(
            company="Razorpay",
            trigger_events=[TriggerEvent(summary="RAY launch", source_quote="q")]))
        monkeypatch.setattr(enricher.cache, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(enricher, "USE_RESEARCH_CACHE", False)

        brief, sources = enricher.enrich_lead(lead)

        assert isinstance(brief, str) and "RAY launch" in brief
        assert isinstance(sources, list) and sources
