"""Business-weighted evaluation: a confusion-cost matrix instead of raw accuracy.

Why this matters more than F1 for routing:

Raw accuracy / macro-F1 implicitly assigns equal cost to every misclassification.
That's wrong here. Routing a billing ticket to engineering wastes ~15 agent-
minutes (the engineer triages and re-routes). Predicting a CRITICAL outage as
LOW costs hours (delayed response, SLA breach, customer churn risk). Treating
those two errors as "one wrong prediction each" hides the entire business
question.

What this module provides:

- `CostMatrix` — a (true x predicted) matrix of costs, in any unit (we use
  agent-minutes). The total cost of a prediction set is just summing the cell
  for each (true, pred) pair.
- Default matrices for category and urgency, grounded in the README's framing:
    * Category misroute costs vary by which team got it (engineering's time is
      more expensive than the catch-all "other" queue).
    * Urgency costs are ASYMMETRIC by ordinal direction: under-escalation
      (predicting LOW when truth is CRITICAL) is much more expensive than
      over-escalation. The asymmetry IS the whole point of using an ordinal
      model in §4.
- `predict_min_cost` — a cost-aware decision rule: instead of argmax(probability),
  pick argmin(expected cost) using the model's predict_proba and the cost matrix.
  This is a real production technique and is what the README means by "tuned
  against this matrix, not raw accuracy" in §5.
- `compare_strategies` — drop-in helper to compare argmax vs cost-tuned vs
  baselines (random, majority) on the same labeled dataset.

The numbers in the default matrices are illustrative but defensible — they
reflect rough operational ratios you'd see in a real B2B support org. In a
real deployment you'd plug in numbers from your own ticketing analytics.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ticketrouting.schemas import Category, Urgency


@dataclass
class CostMatrix:
    """Cost of predicting `pred` when truth is `true`. Rows = true, cols = pred."""

    labels: list[str]
    matrix: np.ndarray  # shape (n, n)
    unit: str = "agent-minutes"
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.labels)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"matrix shape {self.matrix.shape} doesn't match {n} labels"
            )
        if np.any(self.matrix < 0):
            raise ValueError("cost matrix entries must be non-negative")
        self._index = {label: i for i, label in enumerate(self.labels)}

    @classmethod
    def from_dict(
        cls,
        labels: list[str],
        costs: dict[tuple[str, str], float],
        default: float = 0.0,
        unit: str = "agent-minutes",
    ) -> "CostMatrix":
        """Build from sparse {(true, pred): cost}. Unspecified pairs get `default`.

        Diagonal entries (correct predictions) default to 0 unless explicitly set,
        regardless of `default`.
        """
        n = len(labels)
        idx = {label: i for i, label in enumerate(labels)}
        m = np.full((n, n), float(default))
        for i in range(n):
            m[i, i] = 0.0
        for (true, pred), cost in costs.items():
            if true not in idx or pred not in idx:
                raise ValueError(
                    f"cost specified for ({true!r}, {pred!r}) but label not in {labels}"
                )
            m[idx[true], idx[pred]] = float(cost)
        return cls(labels=list(labels), matrix=m, unit=unit)

    def cost_for(self, true_label: str, pred_label: str) -> float:
        return float(self.matrix[self._index[true_label], self._index[pred_label]])

    def total_cost(self, y_true: list[str], y_pred: list[str]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must be the same length")
        return float(sum(self.cost_for(t, p) for t, p in zip(y_true, y_pred)))

    def mean_cost(self, y_true: list[str], y_pred: list[str]) -> float:
        if not y_true:
            return 0.0
        return self.total_cost(y_true, y_pred) / len(y_true)

    def cost_per_1000(self, y_true: list[str], y_pred: list[str]) -> float:
        return self.mean_cost(y_true, y_pred) * 1000.0


# ----------------------------------------------------------------------------
# Default cost matrices
# ----------------------------------------------------------------------------


def default_category_costs() -> CostMatrix:
    """Category routing costs in agent-minutes (rough but defensible numbers).

    Logic:
    - Correct route: 0 (model did its job).
    - Misroute to a specialist team that recognises the error fast: ~10 min
      (they triage, mark, send back).
    - Misroute to a specialist team that may try to fix it before re-routing:
      ~20-30 min (e.g. billing tries to refund what's actually a bug, then
      gives up; engineering investigates a non-engineering issue).
    - Misroute to OTHER: 5 min (it's the catch-all; minimal handling cost).
    - Misroute FROM OTHER: 10 min (re-triage to right team).
    """
    labels = [c.value for c in Category]
    costs: dict[tuple[str, str], float] = {}

    specialised = ["billing", "technical", "account", "bug", "feature_request"]

    for true in specialised:
        for pred in specialised:
            if true == pred:
                continue
            # Default cross-specialist misroute.
            costs[(true, pred)] = 15.0
        # Misrouted INTO the catch-all queue — cheap to spot.
        costs[(true, "other")] = 5.0
        # Misrouted FROM the catch-all to a real team — small handling cost.
        costs[("other", true)] = 10.0

    # Asymmetries that match real operational pain:
    costs[("technical", "billing")] = 30.0       # billing spends time trying to refund a real outage
    costs[("technical", "feature_request")] = 30.0  # product team triages an outage as a feature ask
    costs[("bug", "feature_request")] = 25.0     # eng never sees a real defect because it's parked as "nice to have"
    costs[("feature_request", "bug")] = 5.0      # eng quickly closes it as not-a-bug

    return CostMatrix.from_dict(labels=labels, costs=costs, default=15.0)


def default_urgency_costs() -> CostMatrix:
    """Urgency cost matrix in agent-minutes. ASYMMETRIC by ordinal direction.

    Under-escalation (predicting too low) costs more than over-escalation:
    a missed CRITICAL is an SLA breach with customer-churn risk; an
    over-escalated LOW just wastes some oncall attention.

    Rough scaling (per cell):
    - Correct: 0
    - Over by 1: 5     (oncall briefly looks at it)
    - Over by 2: 15    (real oncall interruption)
    - Over by 3: 30    (full oncall page, no actual incident)
    - Under by 1: 30   (delayed response, mild SLA risk)
    - Under by 2: 120  (significant delay, customer pings again)
    - Under by 3: 480  (8 hours = full SLA breach on a critical)
    """
    labels = [u.value for u in [Urgency.LOW, Urgency.MEDIUM, Urgency.HIGH, Urgency.CRITICAL]]
    rank = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    m = np.zeros((n, n))

    over_costs = {1: 5.0, 2: 15.0, 3: 30.0}
    under_costs = {1: 30.0, 2: 120.0, 3: 480.0}

    for true in labels:
        for pred in labels:
            if true == pred:
                continue
            delta = rank[pred] - rank[true]
            if delta > 0:  # over-escalation
                m[rank[true], rank[pred]] = over_costs[delta]
            else:          # under-escalation
                m[rank[true], rank[pred]] = under_costs[-delta]

    return CostMatrix(labels=labels, matrix=m)


# ----------------------------------------------------------------------------
# Decision rules
# ----------------------------------------------------------------------------


def predict_min_cost(
    proba: np.ndarray, classes: list[str], cost_matrix: CostMatrix
) -> np.ndarray:
    """Cost-aware decision: argmin_pred E[cost | predict pred] using model probs.

    For each row of `proba` (n_samples x n_classes), compute the expected cost
    of predicting each class:
        E[cost(pred=c)] = sum_t P(y=t) * C(t, c)
    and pick the class minimising it. With a SYMMETRIC zero-diagonal cost matrix
    this collapses to argmax(proba); with our asymmetric urgency matrix it
    shifts predictions toward higher urgency when probabilities are close,
    because the cost of under-escalating is steeper.

    `proba` columns must align with `classes`; `cost_matrix.labels` order may
    differ — we re-index internally.
    """
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(
            f"proba shape {proba.shape} doesn't match {len(classes)} classes"
        )

    # Reorder cost matrix to align with `classes` ordering.
    label_to_idx = {label: i for i, label in enumerate(cost_matrix.labels)}
    missing = [c for c in classes if c not in label_to_idx]
    if missing:
        raise ValueError(f"classes {missing} not in cost matrix labels {cost_matrix.labels}")

    perm = np.array([label_to_idx[c] for c in classes])
    C = cost_matrix.matrix[np.ix_(perm, perm)]  # shape (n_classes, n_classes), rows=true, cols=pred

    # Expected cost: proba @ C  -> shape (n_samples, n_classes)
    expected = proba @ C
    pred_idx = np.argmin(expected, axis=1)
    return np.array([classes[i] for i in pred_idx])


# ----------------------------------------------------------------------------
# Baseline strategies
# ----------------------------------------------------------------------------


def random_predictions(y_true: list[str], labels: list[str], seed: int = 42) -> list[str]:
    """Uniform random predictions over `labels`. The 'do nothing' baseline."""
    rng = random.Random(seed)
    return [rng.choice(labels) for _ in y_true]


def majority_predictions(y_true: list[str]) -> list[str]:
    """Always predict the most-common true label in the dataset."""
    if not y_true:
        return []
    most_common = Counter(y_true).most_common(1)[0][0]
    return [most_common] * len(y_true)


def perfect_predictions(y_true: list[str]) -> list[str]:
    """Trivial helper for sanity tests — copy the labels."""
    return list(y_true)


# ----------------------------------------------------------------------------
# Strategy comparison
# ----------------------------------------------------------------------------


@dataclass
class StrategyReport:
    name: str
    total_cost: float
    cost_per_1000: float
    accuracy: float
    n: int

    def pretty(self, unit: str, baseline_cost_per_1000: float | None = None) -> str:
        line = (
            f"{self.name:30s} "
            f"cost/1000={self.cost_per_1000:>10,.1f} {unit}   "
            f"acc={self.accuracy:6.2%}"
        )
        if baseline_cost_per_1000 is not None and baseline_cost_per_1000 > 0:
            reduction = 1 - (self.cost_per_1000 / baseline_cost_per_1000)
            line += f"   vs baseline: {reduction:+7.1%}"
        return line


def compare_strategies(
    y_true: list[str],
    predictions: dict[str, list[str]],
    cost_matrix: CostMatrix,
    baseline_name: str = "random",
) -> list[StrategyReport]:
    """Score each named prediction strategy; return reports in dict order.

    If `baseline_name` is in `predictions`, each report's `pretty()` can format
    the cost reduction relative to that baseline.
    """
    reports: list[StrategyReport] = []
    for name, preds in predictions.items():
        if len(preds) != len(y_true):
            raise ValueError(
                f"strategy {name!r} has {len(preds)} predictions for {len(y_true)} labels"
            )
        accuracy = float(np.mean([p == t for p, t in zip(preds, y_true)])) if y_true else 0.0
        reports.append(
            StrategyReport(
                name=name,
                total_cost=cost_matrix.total_cost(y_true, preds),
                cost_per_1000=cost_matrix.cost_per_1000(y_true, preds),
                accuracy=accuracy,
                n=len(y_true),
            )
        )
    return reports
