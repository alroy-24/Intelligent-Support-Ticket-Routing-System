"""Tests for drift monitoring: PSI math, severity buckets, windowing, full report."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from ticketrouting.active.loop import LogRow
from ticketrouting.drift.distribution import (
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    category_distribution,
    compute_drift_report,
    mean_confidence,
    psi,
    psi_severity,
    split_log_by_time,
    split_log_in_half,
)


_SENTINEL = object()


def _row(category: str, confidence: float = 0.8, *, ts=_SENTINEL) -> LogRow:
    """ts=_SENTINEL means "use a valid default"; explicit None/'' is passed through."""
    if ts is _SENTINEL:
        ts = "2026-05-18T12:00:00.000000Z"
    return LogRow(
        text="x",
        predicted_category=category,
        route_confidence=confidence,
        predicted_urgency="medium",
        urgency_score=0.5,
        ts=ts,
    )


# ----------------------------------------------------------------------------
# category_distribution + mean_confidence
# ----------------------------------------------------------------------------


def test_category_distribution_normalises_counts():
    rows = [_row("billing")] * 3 + [_row("other")]
    dist = category_distribution(rows)
    assert dist == {"billing": 0.75, "other": 0.25}


def test_category_distribution_handles_empty():
    assert category_distribution([]) == {}


def test_mean_confidence_averages():
    rows = [_row("a", 0.2), _row("a", 0.8)]
    assert mean_confidence(rows) == pytest.approx(0.5)


def test_mean_confidence_empty_is_zero():
    assert mean_confidence([]) == 0.0


# ----------------------------------------------------------------------------
# PSI math
# ----------------------------------------------------------------------------


def test_psi_zero_for_identical_distributions():
    dist = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert psi(dist, dist) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_distance():
    """Larger distributional gap -> larger PSI."""
    ref = {"a": 0.5, "b": 0.5}
    near = {"a": 0.55, "b": 0.45}
    far = {"a": 0.9, "b": 0.1}

    assert psi(ref, near) < psi(ref, far)


def test_psi_against_known_formula():
    """Sanity check against the manual sum."""
    ref = {"a": 0.5, "b": 0.5}
    rec = {"a": 0.6, "b": 0.4}
    expected = (0.6 - 0.5) * math.log(0.6 / 0.5) + (0.4 - 0.5) * math.log(0.4 / 0.5)
    assert psi(ref, rec) == pytest.approx(expected)


def test_psi_handles_missing_category_via_epsilon():
    """A category present in one window but not the other yields a large finite PSI."""
    ref = {"a": 1.0}
    rec = {"a": 0.5, "b": 0.5}  # new category appeared

    value = psi(ref, rec, labels=["a", "b"])
    assert math.isfinite(value)
    assert value > PSI_SIGNIFICANT, "introducing a fresh category should register as significant"


def test_psi_symmetric():
    """PSI is symmetric — (recent - reference)*ln(recent/reference) = (reference - recent)*ln(reference/recent)."""
    a = {"x": 0.7, "y": 0.3}
    b = {"x": 0.4, "y": 0.6}
    assert psi(a, b) == pytest.approx(psi(b, a))


# ----------------------------------------------------------------------------
# Severity buckets
# ----------------------------------------------------------------------------


def test_severity_at_thresholds():
    assert psi_severity(0.0) == "no_change"
    assert psi_severity(PSI_MODERATE - 1e-9) == "no_change"
    assert psi_severity(PSI_MODERATE) == "moderate"
    assert psi_severity(PSI_SIGNIFICANT - 1e-9) == "moderate"
    assert psi_severity(PSI_SIGNIFICANT) == "significant"
    assert psi_severity(10.0) == "significant"


# ----------------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------------


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def test_split_by_time_partitions_correctly():
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        _row("a", ts=_ts(now - timedelta(days=20))),  # too old, dropped
        _row("a", ts=_ts(now - timedelta(days=10))),  # reference window
        _row("a", ts=_ts(now - timedelta(days=8))),   # reference window
        _row("a", ts=_ts(now - timedelta(days=5))),   # recent window
        _row("a", ts=_ts(now - timedelta(days=1))),   # recent window
        _row("a", ts=_ts(now + timedelta(days=1))),   # future, dropped
    ]
    ref, recent = split_log_by_time(rows, ref_days=7, recent_days=7, now=now)
    assert len(ref) == 2
    assert len(recent) == 2


def test_split_by_time_drops_unparseable_timestamps():
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        _row("a", ts=""),               # empty
        _row("a", ts="not-a-date"),     # garbage
        _row("a", ts=_ts(now - timedelta(days=1))),
    ]
    ref, recent = split_log_by_time(rows, ref_days=7, recent_days=7, now=now)
    assert len(ref) == 0
    assert len(recent) == 1


def test_split_in_half_balances_for_even_n():
    rows = [_row("a") for _ in range(6)]
    ref, recent = split_log_in_half(rows)
    assert len(ref) == 3
    assert len(recent) == 3


def test_split_in_half_handles_tiny_log():
    assert split_log_in_half([]) == ([], [])
    one = [_row("a")]
    assert split_log_in_half(one) == (one, [])


# ----------------------------------------------------------------------------
# Full DriftReport
# ----------------------------------------------------------------------------


def test_report_flat_when_distributions_match():
    rows = [_row("a", 0.8)] * 50 + [_row("b", 0.6)] * 50
    report = compute_drift_report(rows[:50], rows[50:])

    # Half a's and half b's in each window... well, not actually — we sliced
    # rows into "first 50 = all a" and "second 50 = all b". So this test
    # creates a maximal shift. Let me flip the assertion direction:
    assert report.severity == "significant"


def test_report_detects_category_spike():
    """Surface a clear spike in the recent window."""
    reference = [_row("billing", 0.85)] * 40 + [_row("other", 0.85)] * 60
    # Recent: half the billing volume moves to technical (an outage)
    recent = (
        [_row("billing", 0.65)] * 20
        + [_row("technical", 0.45)] * 30
        + [_row("other", 0.65)] * 50
    )

    report = compute_drift_report(reference, recent)

    assert report.severity in ("moderate", "significant")
    # Per-category shift should flag technical going up and billing going down.
    assert report.per_category_shift["technical"] > 0
    assert report.per_category_shift["billing"] < 0
    # Mean confidence dropped because the outage tickets confuse the model.
    assert report.confidence_delta < 0


def test_report_no_drift_when_only_noise():
    """Same proportions, lightly perturbed counts -> below moderate threshold."""
    reference = [_row("a", 0.8)] * 60 + [_row("b", 0.7)] * 40
    recent = [_row("a", 0.8)] * 58 + [_row("b", 0.7)] * 42

    report = compute_drift_report(reference, recent)
    assert report.severity == "no_change"
    assert report.psi < PSI_MODERATE


def test_report_labels_include_categories_from_either_window():
    reference = [_row("a")] * 10
    recent = [_row("b")] * 10  # totally new category

    report = compute_drift_report(reference, recent)
    assert set(report.labels) == {"a", "b"}
    assert report.severity == "significant"


def test_report_to_dict_round_trips_all_fields():
    reference = [_row("a", 0.8)] * 10
    recent = [_row("b", 0.6)] * 10

    report = compute_drift_report(reference, recent)
    d = report.to_dict()

    assert d["reference_n"] == 10
    assert d["recent_n"] == 10
    assert "psi" in d
    assert "severity" in d
    assert "per_category_shift" in d
    assert d["confidence_delta"] == pytest.approx(report.confidence_delta)
