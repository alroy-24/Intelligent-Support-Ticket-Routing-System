"""Active learning loop: log predictions, surface uncertain ones, merge corrections.

The data flywheel the README describes:

  prod traffic --> /route --> PredictionLog (JSONL, append-only)
                                       |
                                       v
                       surface --threshold 0.6 --max 100
                                       |
                                       v
                       review_queue.jsonl  (human edits human_label)
                                       |
                                       v
                       merge --> reviewed.parquet (text, category)
                                       |
                                       v
                  next training run consumes reviewed.parquet alongside Bitext/Twitter

Why this design:
- JSONL append-only logs survive crashes and concurrent writes (one line = one
  atomic syscall on POSIX; on Windows we accept best-effort ordering). No DB.
- The review file is also JSONL so a human can edit it in vim / a spreadsheet
  / a notebook — no custom tool needed.
- The merge step writes parquet so the corrected rows slot straight into the
  training pipeline that already reads parquets.
- Each LogRow carries the model id so we can tell, post-hoc, which model
  produced which low-confidence cases (drift signal in disguise).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd


# ----------------------------------------------------------------------------
# Schemas (deliberately plain dicts on the wire; dataclasses on the inside)
# ----------------------------------------------------------------------------


@dataclass
class LogRow:
    """One production prediction, as written to the prediction log."""

    text: str
    predicted_category: str
    route_confidence: float
    predicted_urgency: str
    urgency_score: float
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )
    ticket_id: str | None = None
    model_id: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "ts": self.ts,
                "text": self.text,
                "predicted_category": self.predicted_category,
                "route_confidence": self.route_confidence,
                "predicted_urgency": self.predicted_urgency,
                "urgency_score": self.urgency_score,
                "ticket_id": self.ticket_id,
                "model_id": self.model_id,
            }
        )

    @classmethod
    def from_dict(cls, data: dict) -> "LogRow":
        return cls(
            text=data["text"],
            predicted_category=data["predicted_category"],
            route_confidence=float(data["route_confidence"]),
            predicted_urgency=data["predicted_urgency"],
            urgency_score=float(data["urgency_score"]),
            ts=data.get("ts", ""),
            ticket_id=data.get("ticket_id"),
            model_id=data.get("model_id"),
        )


@dataclass
class ReviewRow:
    """One ticket surfaced for human review. The human fills in human_label."""

    id: str                  # stable hash so repeated surfacing of the same row is idempotent
    text: str
    predicted_category: str
    confidence: float
    ts: str
    human_label: str | None = None  # null = unreviewed; "discard" = junk; else the corrected category

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "text": self.text,
                "predicted_category": self.predicted_category,
                "confidence": self.confidence,
                "ts": self.ts,
                "human_label": self.human_label,
            }
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewRow":
        return cls(
            id=data["id"],
            text=data["text"],
            predicted_category=data["predicted_category"],
            confidence=float(data["confidence"]),
            ts=data.get("ts", ""),
            human_label=data.get("human_label"),
        )


# ----------------------------------------------------------------------------
# Prediction log (JSONL, append-only)
# ----------------------------------------------------------------------------


class PredictionLog:
    """Append-only JSONL writer + reader for production predictions."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: LogRow) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(row.to_json() + "\n")

    def __iter__(self) -> Iterator[LogRow]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield LogRow.from_dict(json.loads(line))

    def __len__(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


# ----------------------------------------------------------------------------
# Surfacing low-confidence predictions
# ----------------------------------------------------------------------------


def _stable_id(text: str, ts: str) -> str:
    """Deterministic short id so the same row appears with the same id across runs."""
    return hashlib.sha256((text + "|" + ts).encode()).hexdigest()[:16]


def surface_low_confidence(
    log: PredictionLog,
    threshold: float = 0.6,
    max_n: int | None = 100,
) -> list[ReviewRow]:
    """Return the N rows from `log` with route_confidence strictly below `threshold`.

    Sorted ascending by confidence so the most uncertain comes first — that's the
    most informative example for human relabelling. Ties broken by timestamp
    (older first; the model has had longer to be wrong about them).
    """
    candidates: list[ReviewRow] = []
    for row in log:
        if row.route_confidence >= threshold:
            continue
        candidates.append(
            ReviewRow(
                id=_stable_id(row.text, row.ts),
                text=row.text,
                predicted_category=row.predicted_category,
                confidence=row.route_confidence,
                ts=row.ts,
            )
        )
    candidates.sort(key=lambda r: (r.confidence, r.ts))
    if max_n is not None:
        candidates = candidates[:max_n]
    return candidates


def write_review_queue(rows: list[ReviewRow], path: Path | str) -> Path:
    """Dump review rows as JSONL the human will edit in place."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.to_json() + "\n")
    return out


def read_review_queue(path: Path | str) -> list[ReviewRow]:
    """Read a (possibly human-edited) review JSONL back into ReviewRows."""
    rows: list[ReviewRow] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(ReviewRow.from_dict(json.loads(line)))
    return rows


# ----------------------------------------------------------------------------
# Merging human corrections back into training data
# ----------------------------------------------------------------------------


def apply_corrections(
    corrections_path: Path | str,
    reviewed_parquet_path: Path | str,
    valid_categories: set[str] | None = None,
) -> int:
    """Read a human-edited review JSONL, append accepted corrections to a parquet.

    "Accepted" = human_label is non-null AND not "discard" AND (if valid_categories
    is set) is a member of that set.

    Returns the number of new rows appended. Idempotent: rows are keyed by (id);
    re-running on the same corrections file won't double-count.
    """
    incoming = read_review_queue(corrections_path)
    accepted: list[dict] = []
    for row in incoming:
        label = row.human_label
        if label is None or label == "discard":
            continue
        if valid_categories is not None and label not in valid_categories:
            raise ValueError(
                f"row {row.id!r} has unknown label {label!r}; "
                f"must be one of {sorted(valid_categories)} or 'discard' or null"
            )
        accepted.append(
            {
                "id": row.id,
                "text": row.text,
                "category": label,
                "source": "human_review",
                "reviewed_ts": row.ts,
            }
        )

    if not accepted:
        return 0

    new_df = pd.DataFrame(accepted)
    target = Path(reviewed_parquet_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        existing = pd.read_parquet(target)
        existing_ids = set(existing["id"]) if "id" in existing.columns else set()
        new_df = new_df[~new_df["id"].isin(existing_ids)]
        if new_df.empty:
            return 0
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(target, index=False)
    return len(new_df)
