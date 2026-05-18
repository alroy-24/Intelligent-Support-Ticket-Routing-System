"""Tests for the business-weighted cost evaluation."""
from __future__ import annotations

import numpy as np
import pytest

from ticketrouting.eval.cost import (
    CostMatrix,
    compare_strategies,
    default_category_costs,
    default_urgency_costs,
    majority_predictions,
    perfect_predictions,
    predict_min_cost,
    random_predictions,
)
from ticketrouting.schemas import Category, Urgency


# ----------------------------------------------------------------------------
# CostMatrix construction + lookup
# ----------------------------------------------------------------------------


def test_from_dict_zeroes_diagonal_by_default():
    cm = CostMatrix.from_dict(labels=["a", "b"], costs={}, default=10.0)

    assert cm.cost_for("a", "a") == 0.0
    assert cm.cost_for("b", "b") == 0.0
    assert cm.cost_for("a", "b") == 10.0


def test_from_dict_explicit_overrides_default():
    cm = CostMatrix.from_dict(
        labels=["a", "b", "c"],
        costs={("a", "b"): 100, ("b", "c"): 50},
        default=1.0,
    )

    assert cm.cost_for("a", "b") == 100.0
    assert cm.cost_for("b", "c") == 50.0
    assert cm.cost_for("a", "c") == 1.0  # default applied


def test_from_dict_rejects_unknown_label():
    with pytest.raises(ValueError, match="not in"):
        CostMatrix.from_dict(labels=["a"], costs={("a", "z"): 1.0})


def test_from_dict_rejects_negative_costs():
    with pytest.raises(ValueError, match="non-negative"):
        CostMatrix.from_dict(labels=["a", "b"], costs={("a", "b"): -1.0})


def test_init_rejects_wrong_shape():
    with pytest.raises(ValueError, match="doesn't match"):
        CostMatrix(labels=["a", "b"], matrix=np.zeros((3, 3)))


# ----------------------------------------------------------------------------
# Cost aggregation
# ----------------------------------------------------------------------------


def test_total_cost_sums_each_pair():
    cm = CostMatrix.from_dict(
        labels=["a", "b"], costs={("a", "b"): 10, ("b", "a"): 5}
    )
    y_true = ["a", "a", "b", "a"]
    y_pred = ["a", "b", "a", "b"]

    # cells: 0, 10, 5, 10 -> total 25
    assert cm.total_cost(y_true, y_pred) == 25.0


def test_mean_and_per_1000_cost():
    cm = CostMatrix.from_dict(labels=["a", "b"], costs={("a", "b"): 8.0})
    y_true = ["a", "a", "a", "a"]
    y_pred = ["a", "b", "b", "a"]  # 2 misroutes @ 8 each = 16

    assert cm.total_cost(y_true, y_pred) == 16.0
    assert cm.mean_cost(y_true, y_pred) == 4.0
    assert cm.cost_per_1000(y_true, y_pred) == 4000.0


def test_total_cost_rejects_mismatched_lengths():
    cm = CostMatrix.from_dict(labels=["a"], costs={})
    with pytest.raises(ValueError, match="same length"):
        cm.total_cost(["a"], ["a", "a"])


def test_empty_inputs_give_zero_cost():
    cm = CostMatrix.from_dict(labels=["a"], costs={})
    assert cm.total_cost([], []) == 0.0
    assert cm.mean_cost([], []) == 0.0


# ----------------------------------------------------------------------------
# Default matrices
# ----------------------------------------------------------------------------


def test_default_category_costs_zero_on_correct_routing():
    cm = default_category_costs()
    for label in cm.labels:
        assert cm.cost_for(label, label) == 0.0


def test_default_category_costs_punishes_technical_to_billing_more_than_reverse():
    """A real outage routed to billing wastes more time than a billing question routed to eng."""
    cm = default_category_costs()
    assert cm.cost_for("technical", "billing") > cm.cost_for("billing", "technical")


def test_default_category_costs_other_misroute_is_cheap():
    cm = default_category_costs()
    # Specialist -> OTHER is cheaper than specialist -> wrong specialist.
    for true in ["billing", "technical", "account", "bug"]:
        wrong_specialist = "feature_request" if true != "feature_request" else "bug"
        assert cm.cost_for(true, "other") < cm.cost_for(true, wrong_specialist), (
            f"{true} -> other ({cm.cost_for(true, 'other')}) should be cheaper than "
            f"{true} -> {wrong_specialist} ({cm.cost_for(true, wrong_specialist)})"
        )


def test_default_urgency_costs_under_escalation_more_expensive_than_over():
    """The whole point of the asymmetric matrix. Distance 1 in each direction:
    over (medium predicted as high) << under (high predicted as medium).
    """
    cm = default_urgency_costs()
    over_one = cm.cost_for(Urgency.MEDIUM.value, Urgency.HIGH.value)
    under_one = cm.cost_for(Urgency.HIGH.value, Urgency.MEDIUM.value)
    assert under_one > over_one


def test_default_urgency_costs_worst_case_is_critical_predicted_as_low():
    cm = default_urgency_costs()
    worst = cm.matrix.max()
    actual_worst_cell = cm.cost_for(Urgency.CRITICAL.value, Urgency.LOW.value)
    assert actual_worst_cell == worst, (
        "predicting LOW when truth is CRITICAL should be the most expensive cell"
    )


def test_default_urgency_costs_scale_with_distance():
    """Costs grow with ordinal distance in both directions."""
    cm = default_urgency_costs()
    labels = [Urgency.LOW.value, Urgency.MEDIUM.value, Urgency.HIGH.value, Urgency.CRITICAL.value]
    # Over-escalation: from LOW
    over_1 = cm.cost_for(labels[0], labels[1])
    over_2 = cm.cost_for(labels[0], labels[2])
    over_3 = cm.cost_for(labels[0], labels[3])
    assert over_1 < over_2 < over_3

    # Under-escalation: from CRITICAL
    under_1 = cm.cost_for(labels[3], labels[2])
    under_2 = cm.cost_for(labels[3], labels[1])
    under_3 = cm.cost_for(labels[3], labels[0])
    assert under_1 < under_2 < under_3


# ----------------------------------------------------------------------------
# predict_min_cost decision rule
# ----------------------------------------------------------------------------


def test_min_cost_equals_argmax_on_symmetric_zero_diagonal_matrix():
    """When cost is symmetric and equal off-diagonal, min-cost == argmax(proba)."""
    classes = ["a", "b", "c"]
    cm = CostMatrix.from_dict(labels=classes, costs={}, default=1.0)
    proba = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
        ]
    )

    preds = predict_min_cost(proba, classes, cm)

    assert list(preds) == ["a", "b", "c"]


def test_min_cost_shifts_toward_higher_urgency_when_uncertain():
    """With the asymmetric urgency matrix, a tossup between HIGH and CRITICAL should pick CRITICAL.

    Under-escalation is much more expensive, so the expected cost of predicting
    HIGH (and being wrong about a CRITICAL) is higher than predicting CRITICAL
    (and being wrong about a HIGH).
    """
    cm = default_urgency_costs()
    classes = cm.labels  # ["low", "medium", "high", "critical"]
    # 50/50 between high and critical:
    proba = np.array([[0.0, 0.0, 0.5, 0.5]])

    preds = predict_min_cost(proba, classes, cm)

    assert preds[0] == "critical"


def test_min_cost_handles_reordered_classes():
    """Cost matrix label order need not match proba column order."""
    # default=10 so picking the third class `c` isn't trivially free.
    cm = CostMatrix.from_dict(
        labels=["a", "b", "c"],
        costs={("a", "b"): 100.0, ("b", "a"): 1.0},
        default=10.0,
    )
    # proba columns in a different order:
    classes = ["c", "b", "a"]
    proba = np.array([[0.0, 0.5, 0.5]])  # 50/50 between b and a

    preds = predict_min_cost(proba, classes, cm)

    # Predicting a when truth is b costs 1; predicting b when truth is a costs 100.
    # So predicting a is the safer bet (E[cost]=0.5 vs 50.0).
    assert preds[0] == "a"


def test_min_cost_rejects_classes_not_in_matrix():
    cm = CostMatrix.from_dict(labels=["a", "b"], costs={})
    proba = np.zeros((1, 2))
    with pytest.raises(ValueError, match="not in cost matrix"):
        predict_min_cost(proba, ["a", "z"], cm)


def test_min_cost_rejects_proba_class_mismatch():
    cm = CostMatrix.from_dict(labels=["a", "b"], costs={})
    proba = np.zeros((1, 3))
    with pytest.raises(ValueError, match="doesn't match"):
        predict_min_cost(proba, ["a", "b"], cm)


# ----------------------------------------------------------------------------
# Baselines + comparison
# ----------------------------------------------------------------------------


def test_perfect_predictor_zero_cost():
    cm = default_category_costs()
    y_true = ["billing", "technical", "account", "other"]

    assert cm.total_cost(y_true, perfect_predictions(y_true)) == 0.0


def test_random_predictions_are_deterministic_for_same_seed():
    labels = [c.value for c in Category]
    a = random_predictions(["x"] * 50, labels=labels, seed=7)
    b = random_predictions(["x"] * 50, labels=labels, seed=7)
    assert a == b


def test_majority_predictions_uses_most_common_label():
    y = ["billing"] * 8 + ["account"] * 2
    assert majority_predictions(y) == ["billing"] * 10


def test_compare_strategies_reports_each_strategy():
    cm = CostMatrix.from_dict(labels=["a", "b"], costs={("a", "b"): 4.0, ("b", "a"): 1.0})
    y_true = ["a", "a", "b", "b"]
    reports = compare_strategies(
        y_true,
        predictions={
            "perfect": ["a", "a", "b", "b"],
            "always_a": ["a", "a", "a", "a"],
            "always_b": ["b", "b", "b", "b"],
        },
        cost_matrix=cm,
    )

    by_name = {r.name: r for r in reports}
    assert by_name["perfect"].total_cost == 0.0
    assert by_name["always_a"].total_cost == 2.0  # two b's predicted as a, cost 1 each
    assert by_name["always_b"].total_cost == 8.0  # two a's predicted as b, cost 4 each
    assert by_name["perfect"].accuracy == 1.0
    assert by_name["always_a"].accuracy == 0.5


def test_compare_strategies_rejects_length_mismatch():
    cm = CostMatrix.from_dict(labels=["a"], costs={})
    with pytest.raises(ValueError, match="predictions for"):
        compare_strategies(["a"], {"bad": ["a", "a"]}, cm)


def test_model_with_cost_tuned_rule_beats_random_on_default_matrix():
    """End-to-end: cost-tuned predictions on a noisy proba beat random predictions."""
    rng = np.random.default_rng(0)
    cm = default_category_costs()
    classes = cm.labels

    n = 200
    # 80% billing, 20% other — simulating a Bitext-style skew.
    y_true = (["billing"] * (n * 4 // 5)) + (["other"] * (n - n * 4 // 5))

    # Noisy proba that mostly puts mass on the true class.
    proba = np.zeros((n, len(classes)))
    for i, t in enumerate(y_true):
        proba[i, classes.index(t)] = 0.7
        # spread remaining 0.3 over the others
        for j, c in enumerate(classes):
            if c != t:
                proba[i, j] = 0.3 / (len(classes) - 1)
        # tiny perturbation
        proba[i] += rng.normal(0, 0.02, len(classes))
        proba[i] = np.clip(proba[i], 0, None)
        proba[i] /= proba[i].sum()

    cost_tuned = predict_min_cost(proba, classes, cm).tolist()
    random_route = random_predictions(y_true, labels=classes, seed=0)

    assert cm.total_cost(y_true, cost_tuned) < cm.total_cost(y_true, random_route)
