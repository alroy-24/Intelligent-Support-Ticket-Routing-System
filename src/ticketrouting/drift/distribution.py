"""Drift detection on the production PredictionLog.

What this catches:

- **Category-distribution drift** via the Population Stability Index (PSI).
  PSI is the standard categorical-drift metric in banking / credit risk;
  conventional thresholds are PSI<0.10 stable, 0.10–0.25 moderate, >=0.25
  significant. PSI is direction-agnostic: a category that spikes and one
  that collapses both register.
- **Mean-confidence drift** as a secondary signal. A drop in average
  route_confidence is a fast canary for input distribution shift even
  when the predicted-label distribution looks stable (the model is
  becoming less sure of itself across the board).
- **Per-category point-shift** (recent_prop - reference_prop) so the
  operator can see WHICH category moved, not just that something did.
  That's the actionable difference between "noticed drift" and "found
  the outage".

What this does NOT do (deliberate scope cut for v1):

- Embedding drift. The README mentions it, but storing embeddings in the
  log doubles its size and requires a schema change. Future work.
- Concept drift on labels. Without ground truth from production we can't
  measure label drift directly; the active-learning loop's reviewed.parquet
  is the closest substitute and is tracked separately.

Windowing:

- `split_log_by_time(rows, ref_days, recent_days)` for the realistic case
  where the log spans weeks. Reference window = the ref_days *before* the
  recent_days. So "last 7 days vs the prior 7 days" is the natural cron
  setting.
- `split_log_in_half(rows)` for the unrealistic-but-useful case of a fresh
  log that hasn't accumulated time-separated traffic yet (and for tests).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ticketrouting.active.loop import LogRow

# PSI severity thresholds — the industry-conventional buckets.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


# ----------------------------------------------------------------------------
# Distribution + PSI math
# ----------------------------------------------------------------------------


def category_distribution(rows: list[LogRow]) -> dict[str, float]:
    """Empirical category proportions over `rows`. Empty rows -> {}."""
    if not rows:
        return {}
    counts = Counter(r.predicted_category for r in rows)
    total = sum(counts.values())
    return {cat: n / total for cat, n in counts.items()}


def mean_confidence(rows: list[LogRow]) -> float:
    """Mean route_confidence across `rows`. Empty rows -> 0.0."""
    if not rows:
        return 0.0
    return sum(r.route_confidence for r in rows) / len(rows)


def psi(
    reference: dict[str, float],
    recent: dict[str, float],
    labels: list[str] | None = None,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index.

    PSI = sum_c (recent_c - reference_c) * ln(recent_c / reference_c)

    `labels` controls which categories to score over. If None, the union of
    keys in both distributions is used. `epsilon` floors zero proportions so
    log() is defined when a category appears in one window but not the other
    (which is the exact case PSI is designed to catch loudly).
    """
    if labels is None:
        labels = sorted(set(reference) | set(recent))

    total = 0.0
    for cat in labels:
        r = max(reference.get(cat, 0.0), epsilon)
        c = max(recent.get(cat, 0.0), epsilon)
        total += (c - r) * math.log(c / r)
    return total


def psi_severity(psi_value: float) -> str:
    """Conventional buckets: no_change | moderate | significant."""
    if psi_value < PSI_MODERATE:
        return "no_change"
    if psi_value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


# ----------------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------------


def _parse_ts(ts: str) -> datetime | None:
    """LogRow.ts is ISO-ish but tolerant of trailing Z + microseconds."""
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def split_log_by_time(
    rows: list[LogRow],
    ref_days: int,
    recent_days: int,
    now: datetime | None = None,
) -> tuple[list[LogRow], list[LogRow]]:
    """Partition by timestamp into (reference, recent) windows.

    Reference window = (now - ref_days - recent_days, now - recent_days].
    Recent window    = (now - recent_days, now].

    Rows older than the reference window are silently dropped; rows newer than
    `now` (clock skew, future-dated tests) are silently dropped from recent
    and not added to reference. Rows with unparseable timestamps are dropped.
    """
    now = now or datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=recent_days)
    ref_cutoff = recent_cutoff - timedelta(days=ref_days)

    reference: list[LogRow] = []
    recent: list[LogRow] = []
    for row in rows:
        ts = _parse_ts(row.ts)
        if ts is None or ts > now:
            continue
        if ts > recent_cutoff:
            recent.append(row)
        elif ts > ref_cutoff:
            reference.append(row)
    return reference, recent


def split_log_in_half(rows: list[LogRow]) -> tuple[list[LogRow], list[LogRow]]:
    """Split by row order. Useful when the log hasn't spanned the time windows yet."""
    n = len(rows)
    if n < 2:
        return rows, []
    mid = n // 2
    return rows[:mid], rows[mid:]


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


@dataclass
class DriftReport:
    reference_n: int
    recent_n: int
    reference_distribution: dict[str, float]
    recent_distribution: dict[str, float]
    psi: float
    severity: str
    per_category_shift: dict[str, float]
    mean_confidence_reference: float
    mean_confidence_recent: float
    confidence_delta: float
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reference_n": self.reference_n,
            "recent_n": self.recent_n,
            "reference_distribution": self.reference_distribution,
            "recent_distribution": self.recent_distribution,
            "psi": self.psi,
            "severity": self.severity,
            "per_category_shift": self.per_category_shift,
            "mean_confidence_reference": self.mean_confidence_reference,
            "mean_confidence_recent": self.mean_confidence_recent,
            "confidence_delta": self.confidence_delta,
            "labels": self.labels,
        }

    def pretty(self) -> str:
        lines = [
            f"Reference window: n={self.reference_n:,}",
            f"Recent window:    n={self.recent_n:,}",
            "",
            f"PSI:              {self.psi:.4f}  ({self.severity})",
            f"Mean confidence:  {self.mean_confidence_reference:.3f} -> "
            f"{self.mean_confidence_recent:.3f}  (delta {self.confidence_delta:+.3f})",
            "",
            "Per-category shift (recent - reference):",
        ]
        for cat in self.labels:
            ref = self.reference_distribution.get(cat, 0.0)
            rec = self.recent_distribution.get(cat, 0.0)
            shift = rec - ref
            marker = ""
            if abs(shift) >= 0.10:
                marker = "  <- significant shift"
            lines.append(
                f"  {cat:18s} {ref:6.1%} -> {rec:6.1%}   {shift:+.1%}{marker}"
            )
        return "\n".join(lines)


def compute_drift_report(
    reference: list[LogRow],
    recent: list[LogRow],
) -> DriftReport:
    """Build a DriftReport from two pre-windowed row lists."""
    ref_dist = category_distribution(reference)
    rec_dist = category_distribution(recent)
    labels = sorted(set(ref_dist) | set(rec_dist))

    psi_value = psi(ref_dist, rec_dist, labels=labels)
    severity = psi_severity(psi_value)
    per_cat_shift = {cat: rec_dist.get(cat, 0.0) - ref_dist.get(cat, 0.0) for cat in labels}

    ref_conf = mean_confidence(reference)
    rec_conf = mean_confidence(recent)

    return DriftReport(
        reference_n=len(reference),
        recent_n=len(recent),
        reference_distribution=ref_dist,
        recent_distribution=rec_dist,
        psi=psi_value,
        severity=severity,
        per_category_shift=per_cat_shift,
        mean_confidence_reference=ref_conf,
        mean_confidence_recent=rec_conf,
        confidence_delta=rec_conf - ref_conf,
        labels=labels,
    )
