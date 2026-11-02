"""Display parsing of the enricher's raw source strings."""

from agents.sources import clean_snippet, parse_source

RAW = """Indian Startup IPO Tracker 2026 · Sun, 30 Aug 2026 07:00:54 GMT
![logo](https://x.com/a.png) Image 10: Socure In June 2026, the unicorn filed its **DRHP** with [SEBI](https://sebi.gov.in).
## Related
[https://www.inc42.com/features/ipo-tracker/]"""


def test_splits_title_date_url_and_domain():
    s = parse_source(RAW)
    assert s.title == "Indian Startup IPO Tracker 2026"
    assert s.date == "30 Aug 2026"
    assert s.url == "https://www.inc42.com/features/ipo-tracker/"
    assert s.domain == "inc42.com"


def test_snippet_drops_page_furniture():
    snippet = parse_source(RAW).snippet
    assert "![" not in snippet and "Image 10" not in snippet and "##" not in snippet
    assert "**" not in snippet and "https://" not in snippet
    assert "filed its DRHP with SEBI" in snippet


def test_source_without_header_or_url_still_renders():
    s = parse_source("Just some content with no header")
    assert s.snippet == "Just some content with no header"
    assert s.title == "Untitled result" and s.url == "" and s.date == ""


def test_unparseable_date_is_kept_as_written():
    assert parse_source("T · 03 Sep 2026\nbody\n[https://a.com/x]").date == "03 Sep 2026"


def test_long_snippet_is_cut_at_a_word():
    text = "word " * 100
    out = clean_snippet(text, limit=50)
    assert out.endswith("…") and len(out) <= 51 and not out[:-1].endswith(" ")
