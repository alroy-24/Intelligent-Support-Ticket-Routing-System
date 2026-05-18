"""Tests for the active-learning loop machinery."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ticketrouting.active.loop import (
    LogRow,
    PredictionLog,
    ReviewRow,
    _stable_id,
    apply_corrections,
    read_review_queue,
    surface_low_confidence,
    write_review_queue,
)


# ----------------------------------------------------------------------------
# PredictionLog
# ----------------------------------------------------------------------------


def _row(text: str, conf: float, **kw) -> LogRow:
    defaults = dict(
        text=text,
        predicted_category=kw.pop("predicted_category", "billing"),
        route_confidence=conf,
        predicted_urgency=kw.pop("predicted_urgency", "medium"),
        urgency_score=kw.pop("urgency_score", 0.5),
        ts=kw.pop("ts", "2026-05-18T12:00:00.000000Z"),
    )
    defaults.update(kw)
    return LogRow(**defaults)


def test_append_writes_one_jsonl_line_per_row(tmp_path):
    log = PredictionLog(tmp_path / "preds.jsonl")
    log.append(_row("a", 0.9))
    log.append(_row("b", 0.4))

    with (tmp_path / "preds.jsonl").open() as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 2
    assert lines[0]["text"] == "a"
    assert lines[1]["text"] == "b"


def test_iteration_reads_back_typed_rows(tmp_path):
    log = PredictionLog(tmp_path / "preds.jsonl")
    log.append(_row("hello", 0.55))

    rows = list(log)
    assert len(rows) == 1
    assert rows[0].text == "hello"
    assert rows[0].route_confidence == pytest.approx(0.55)


def test_log_length_is_zero_when_file_missing(tmp_path):
    log = PredictionLog(tmp_path / "never_written.jsonl")
    assert len(log) == 0
    assert list(log) == []


def test_log_ignores_blank_lines(tmp_path):
    path = tmp_path / "p.jsonl"
    log = PredictionLog(path)
    log.append(_row("a", 0.9))
    # Inject a blank line and a trailing newline
    with path.open("a") as f:
        f.write("\n\n")
    log.append(_row("b", 0.4))

    rows = list(log)
    assert [r.text for r in rows] == ["a", "b"]


# ----------------------------------------------------------------------------
# Surface
# ----------------------------------------------------------------------------


def test_surface_drops_high_confidence_rows(tmp_path):
    log = PredictionLog(tmp_path / "p.jsonl")
    log.append(_row("a", 0.9))
    log.append(_row("b", 0.5))
    log.append(_row("c", 0.99))

    surfaced = surface_low_confidence(log, threshold=0.6)
    assert [r.text for r in surfaced] == ["b"]


def test_surface_sorts_by_ascending_confidence(tmp_path):
    log = PredictionLog(tmp_path / "p.jsonl")
    log.append(_row("medium-uncertain", 0.55, ts="2026-05-18T12:00:01Z"))
    log.append(_row("most-uncertain", 0.20, ts="2026-05-18T12:00:02Z"))
    log.append(_row("near-threshold", 0.59, ts="2026-05-18T12:00:03Z"))

    surfaced = surface_low_confidence(log, threshold=0.6)
    assert [r.text for r in surfaced] == [
        "most-uncertain",
        "medium-uncertain",
        "near-threshold",
    ]


def test_surface_breaks_ties_by_older_first(tmp_path):
    log = PredictionLog(tmp_path / "p.jsonl")
    log.append(_row("newer", 0.4, ts="2026-05-18T12:00:02Z"))
    log.append(_row("older", 0.4, ts="2026-05-18T12:00:01Z"))

    surfaced = surface_low_confidence(log, threshold=0.6)
    assert [r.text for r in surfaced] == ["older", "newer"]


def test_surface_respects_max_n(tmp_path):
    log = PredictionLog(tmp_path / "p.jsonl")
    for i in range(20):
        log.append(_row(f"row-{i}", 0.1 + i * 0.01))

    surfaced = surface_low_confidence(log, threshold=0.6, max_n=5)
    assert len(surfaced) == 5
    # Most-uncertain first
    assert surfaced[0].text == "row-0"


def test_surface_ids_are_stable_across_runs(tmp_path):
    log = PredictionLog(tmp_path / "p.jsonl")
    log.append(_row("same text", 0.3, ts="2026-05-18T12:00:00Z"))

    first = surface_low_confidence(log)[0].id
    second = surface_low_confidence(log)[0].id
    assert first == second
    assert first == _stable_id("same text", "2026-05-18T12:00:00Z")


# ----------------------------------------------------------------------------
# Review queue read/write roundtrip
# ----------------------------------------------------------------------------


def test_write_then_read_review_queue_roundtrips(tmp_path):
    rows = [
        ReviewRow(id="abc", text="x", predicted_category="billing", confidence=0.4, ts="t1"),
        ReviewRow(
            id="def", text="y", predicted_category="account", confidence=0.5, ts="t2",
            human_label="bug",
        ),
    ]
    out = tmp_path / "review.jsonl"
    write_review_queue(rows, out)

    back = read_review_queue(out)
    assert len(back) == 2
    assert back[0].id == "abc"
    assert back[0].human_label is None
    assert back[1].human_label == "bug"


# ----------------------------------------------------------------------------
# Merging human corrections
# ----------------------------------------------------------------------------


def test_apply_corrections_skips_unreviewed_and_discarded(tmp_path):
    rows = [
        ReviewRow(id="a", text="x", predicted_category="billing", confidence=0.3, ts="t"),
        ReviewRow(
            id="b", text="y", predicted_category="account", confidence=0.4, ts="t",
            human_label="discard",
        ),
        ReviewRow(
            id="c", text="z", predicted_category="other", confidence=0.5, ts="t",
            human_label="bug",
        ),
    ]
    reviews_path = tmp_path / "rev.jsonl"
    write_review_queue(rows, reviews_path)

    out = tmp_path / "reviewed.parquet"
    n = apply_corrections(reviews_path, out, valid_categories={"bug", "billing"})

    assert n == 1
    df = pd.read_parquet(out)
    assert df.iloc[0]["text"] == "z"
    assert df.iloc[0]["category"] == "bug"
    assert df.iloc[0]["source"] == "human_review"


def test_apply_corrections_rejects_unknown_label(tmp_path):
    rows = [
        ReviewRow(
            id="a", text="x", predicted_category="billing", confidence=0.3, ts="t",
            human_label="nonsense",
        ),
    ]
    reviews_path = tmp_path / "rev.jsonl"
    write_review_queue(rows, reviews_path)
    out = tmp_path / "reviewed.parquet"

    with pytest.raises(ValueError, match="unknown label"):
        apply_corrections(reviews_path, out, valid_categories={"bug"})


def test_apply_corrections_is_idempotent(tmp_path):
    """Re-running the same corrections must not double-count rows."""
    rows = [
        ReviewRow(
            id="a", text="x", predicted_category="billing", confidence=0.3, ts="t",
            human_label="bug",
        ),
    ]
    reviews_path = tmp_path / "rev.jsonl"
    write_review_queue(rows, reviews_path)
    out = tmp_path / "reviewed.parquet"

    n1 = apply_corrections(reviews_path, out, valid_categories={"bug"})
    n2 = apply_corrections(reviews_path, out, valid_categories={"bug"})

    assert n1 == 1
    assert n2 == 0
    df = pd.read_parquet(out)
    assert len(df) == 1


def test_apply_corrections_appends_new_rows_to_existing_parquet(tmp_path):
    out = tmp_path / "reviewed.parquet"

    # First batch
    write_review_queue(
        [
            ReviewRow(
                id="a", text="x", predicted_category="billing", confidence=0.3, ts="t",
                human_label="bug",
            )
        ],
        tmp_path / "rev1.jsonl",
    )
    apply_corrections(tmp_path / "rev1.jsonl", out, valid_categories={"bug", "billing"})

    # Second batch with a new id
    write_review_queue(
        [
            ReviewRow(
                id="b", text="y", predicted_category="other", confidence=0.4, ts="t",
                human_label="billing",
            )
        ],
        tmp_path / "rev2.jsonl",
    )
    n = apply_corrections(tmp_path / "rev2.jsonl", out, valid_categories={"bug", "billing"})

    assert n == 1
    df = pd.read_parquet(out)
    assert sorted(df["id"]) == ["a", "b"]


def test_apply_corrections_returns_zero_when_nothing_accepted(tmp_path):
    write_review_queue([], tmp_path / "rev.jsonl")
    out = tmp_path / "reviewed.parquet"

    n = apply_corrections(tmp_path / "rev.jsonl", out, valid_categories={"bug"})

    assert n == 0
    assert not out.exists()


# ----------------------------------------------------------------------------
# RoutingService logging hook
# ----------------------------------------------------------------------------


def test_routing_service_writes_to_log_when_configured(tmp_path):
    """End-to-end: a /route call appends one LogRow to the configured log."""
    import asyncio
    from dataclasses import dataclass

    from ticketrouting.api.service import RoutingService
    from ticketrouting.schemas import TicketIn

    @dataclass
    class _FakeCatPipe:
        @property
        def classes_(self):
            return ["billing"]

        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.42]] * len(X))

    log_path = tmp_path / "preds.jsonl"
    svc = RoutingService(
        category_pipe=_FakeCatPipe(),
        urgency_pipe=None,
        summarizer=None,
        prediction_log=PredictionLog(log_path),
    )

    asyncio.run(svc.route(TicketIn(text="test ticket", ticket_id="t-1")))

    rows = list(PredictionLog(log_path))
    assert len(rows) == 1
    assert rows[0].text == "test ticket"
    assert rows[0].predicted_category == "billing"
    assert rows[0].route_confidence == pytest.approx(0.42)
    assert rows[0].ticket_id == "t-1"


def test_routing_service_swallows_log_write_failure(tmp_path, monkeypatch):
    """If the log write fails for any reason, the /route response must still succeed."""
    import asyncio
    from dataclasses import dataclass

    from ticketrouting.api.service import RoutingService
    from ticketrouting.schemas import TicketIn

    @dataclass
    class _FakeCatPipe:
        @property
        def classes_(self):
            return ["billing"]

        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.7]] * len(X))

    class _BrokenLog(PredictionLog):
        def append(self, row):
            raise IOError("disk full, etc")

    svc = RoutingService(
        category_pipe=_FakeCatPipe(),
        urgency_pipe=None,
        summarizer=None,
        prediction_log=_BrokenLog(tmp_path / "preds.jsonl"),
    )

    # Should NOT raise
    decision = asyncio.run(svc.route(TicketIn(text="x")))
    assert decision.route_to[0].value == "billing"
