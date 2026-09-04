"""Tests for the CSV log and duplicate detection.

Includes a regression for the second silent bug this project carried: `output_log.csv` was
written without a header row, so `csv.DictReader` treated the first lead as the header. That
made `already_sent()` mis-parse every row, and `--resume` could not have worked.
"""

from tools.logger import FIELDNAMES, already_sent, log_result


def test_creates_log_with_header(tmp_path, lead):
    path = tmp_path / "out.csv"
    log_result(lead, draft="Subject: Hi\n\nBody", status="SENT", log_path=path)

    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.split(",") == FIELDNAMES


def test_header_written_exactly_once_across_appends(tmp_path, lead):
    """Regression: a header per append, or none at all, both corrupt DictReader parsing."""
    path = tmp_path / "out.csv"
    for _ in range(3):
        log_result(lead, draft="d", status="SENT", log_path=path)

    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines[0].startswith("timestamp")
    assert sum(1 for l in lines if l.startswith("timestamp")) == 1
    assert len(lines) == 4  # header + 3 rows


def test_already_sent_round_trips_through_the_log(tmp_path, lead):
    """The bug this guards: without a header the first logged lead was invisible to
    already_sent, so --resume would have re-mailed it."""
    path = tmp_path / "out.csv"
    log_result(lead, draft="d", status="SENT", log_path=path)

    assert already_sent(lead["email"], path)
    assert not already_sent("someone-else@example.com", path)


def test_already_sent_ignores_non_sent_statuses(tmp_path, lead):
    """A held or failed lead must remain eligible on the next run."""
    path = tmp_path / "out.csv"
    log_result(lead, draft="d", status="NEEDS_REVIEW", log_path=path)
    log_result(lead, draft="", status="DRAFT_ERROR: boom", log_path=path)
    log_result(lead, draft="d", status="DRY_RUN", log_path=path)

    assert not already_sent(lead["email"], path)


def test_already_sent_on_missing_file(tmp_path):
    assert not already_sent("a@b.com", tmp_path / "nope.csv")


def test_critique_is_recorded_for_review_rows(tmp_path, lead):
    path = tmp_path / "out.csv"
    log_result(lead, draft="d", status="NEEDS_REVIEW", log_path=path,
               critique="- claim X unsupported\n- claim Y unsupported")

    body = path.read_text(encoding="utf-8")
    assert "claim X unsupported" in body
    # Newlines are flattened so one logical row stays one CSV row.
    assert "\n- claim Y" not in body


def test_draft_preview_is_truncated_and_single_line(tmp_path, lead):
    path = tmp_path / "out.csv"
    log_result(lead, draft="Subject: Hi\n\n" + ("x" * 500), status="SENT", log_path=path)

    import csv
    row = list(csv.DictReader(open(path, encoding="utf-8")))[0]
    assert len(row["draft_preview"]) <= 200
    assert "\n" not in row["draft_preview"]


def test_creates_parent_directory(tmp_path, lead):
    path = tmp_path / "nested" / "dir" / "out.csv"
    log_result(lead, draft="d", status="SENT", log_path=path)
    assert path.exists()
