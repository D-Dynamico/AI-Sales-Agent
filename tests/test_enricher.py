"""Regression tests for the enricher's source extraction.

The headline case: `TavilySearch.invoke()` returns a dict, and the original code did
`for r in results if "content" in r`, which iterates the dict's *keys* and matches nothing.
Every lead therefore reached the LLM with an empty context and the "research" was invented.
The test named for that bug is the one that must never go green-to-red unnoticed.
"""

from agents import enricher
from agents.enricher import NO_RESEARCH, _extract_sources


def test_extracts_content_from_tavily_dict_response(tavily_dict_response):
    """The bug: a dict response yielded zero sources. Guard it explicitly."""
    sources = _extract_sources(tavily_dict_response)

    assert len(sources) == 2
    joined = "\n".join(sources)
    assert "conversational AI account manager" in joined
    assert "Bengaluru enterprise unit" in joined


def test_dict_response_is_not_silently_empty(tavily_dict_response):
    """Directly asserts the failure mode: context must not come back blank."""
    context = "\n\n".join(_extract_sources(tavily_dict_response))
    assert context.strip() != ""
    assert len(context) > 50


def test_extracts_from_legacy_list_response():
    """Older langchain-tavily returned a bare list; both shapes must work."""
    sources = _extract_sources([
        {"content": "Meesho revenue grew 34.5%.", "title": "Meesho FY26", "url": "u"},
    ])
    assert len(sources) == 1
    assert "34.5%" in sources[0]


def test_includes_title_date_and_url_for_traceability(tavily_dict_response):
    """The critic quotes against these snippets, so provenance has to survive."""
    first = _extract_sources(tavily_dict_response)[0]
    assert "Razorpay launches RAY" in first
    assert "03 Sep 2026" in first
    assert "https://example.com/ray" in first


def test_skips_entries_with_no_content():
    sources = _extract_sources({"results": [
        {"content": "", "title": "empty"},
        {"title": "no content key at all"},
        {"content": "   ", "title": "whitespace"},
        {"content": "real text here", "title": "good"},
    ]})
    assert len(sources) == 1
    assert "real text here" in sources[0]


def test_handles_empty_and_malformed_responses():
    assert _extract_sources({}) == []
    assert _extract_sources({"results": []}) == []
    assert _extract_sources([]) == []
    assert _extract_sources(None) == []
    assert _extract_sources({"results": ["a bare string, not a dict"]}) == []


def test_no_sources_short_circuits_without_calling_the_llm(monkeypatch, lead):
    """With nothing to summarise, the model must not be given the chance to invent a brief."""
    monkeypatch.setattr(enricher, "TavilySearch",
                        lambda **kw: type("S", (), {"invoke": lambda self, q: {"results": []}})())

    def explode(*a, **k):
        raise AssertionError("ChatGroq must not be constructed when there are no sources")

    monkeypatch.setattr(enricher, "ChatGroq", explode)

    brief, sources = enricher.enrich_lead(lead)
    assert brief == NO_RESEARCH
    assert sources == []
