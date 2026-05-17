"""Tests for the ordinal urgency baseline.

Strategy: build a tiny synthetic dataset where the ordering of urgency is
literally encoded in the keyword count, then check the trained model:

  1. Returns valid probabilities (sum to 1, non-negative, on the right classes).
  2. Beats the majority-class predictor on MAE — proves it learned the ordering,
     not just the marginal.
  3. Respects the ordinal axis on a clearly-graded test sentence: the model's
     probability mass on Critical should rise as we add more "urgent" tokens.

We don't test exact predictions on noisy text — that's what the real labeled
dataset is for. These tests guard the contract and the ordinal property.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ticketrouting.models.urgency.baseline import (
    URGENCY_ORDER,
    OrdinalLogReg,
    build_pipeline,
    evaluate,
    train_test_eval,
)
from ticketrouting.schemas import Urgency


def _toy_dataset(n_per_class: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic tickets where higher urgency is a strict superset of features.

    Class structure:
      low:      contains sev1
      medium:   contains sev1, sev2
      high:     contains sev1, sev2, sev3
      critical: contains sev1, sev2, sev3, sev4

    This is what an ordinal classifier *should* shine at: the K-1 binary
    thresholds map cleanly onto the presence of sev2 / sev3 / sev4. TF-IDF
    normalisation doesn't hurt because the signal is presence-based, not
    repetition-based.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rank, label in enumerate(URGENCY_ORDER):
        for _ in range(n_per_class):
            severity_tokens = [f"sev{i+1}" for i in range(rank + 1)]
            # A small amount of shared filler so vocab is non-trivial.
            filler = ["filler"] * rng.integers(1, 4)
            rng.shuffle(severity_tokens)
            text = " ".join(severity_tokens + filler)
            rows.append({"text": text, "urgency": label})
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def test_predict_proba_is_well_formed():
    df = _toy_dataset()
    pipe, _ = train_test_eval(df, random_state=0)

    proba = pipe.predict_proba(df["text"].tolist())

    assert proba.shape == (len(df), len(URGENCY_ORDER))
    assert np.all(proba >= 0), "no negative probabilities after clip"
    assert np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_returns_known_classes():
    df = _toy_dataset()
    pipe, _ = train_test_eval(df, random_state=0)

    preds = pipe.predict(df["text"].tolist())

    assert set(preds).issubset(set(URGENCY_ORDER))


def test_beats_majority_baseline_on_mae():
    """MAE must be strictly better than the trivial 'always predict the median' rule.

    With a balanced toy dataset, the optimal constant prediction (medium) gives
    MAE = (1+0+1+2)/4 = 1.0. A trained ordinal model should clearly beat that.
    """
    df = _toy_dataset(n_per_class=40)
    _, report = train_test_eval(df, random_state=0)

    rank_map = {c: i for i, c in enumerate(URGENCY_ORDER)}
    y_true_rank = np.array([rank_map[v] for v in df["urgency"]])
    majority_pred = "medium"
    majority_mae = float(np.abs(y_true_rank - rank_map[majority_pred]).mean())

    assert report.mae < majority_mae, (
        f"ordinal model MAE {report.mae:.3f} should beat majority-class MAE {majority_mae:.3f}"
    )


def test_predicted_rank_rises_with_severity():
    """The argmax prediction should not drop as we add severity tokens."""
    df = _toy_dataset(n_per_class=60)
    pipe, _ = train_test_eval(df, random_state=0)

    rank_map = {c: i for i, c in enumerate(URGENCY_ORDER)}
    ranks = []
    for n in [1, 2, 3, 4]:
        text = " ".join([f"sev{i+1}" for i in range(n)])
        pred = pipe.predict([text])[0]
        ranks.append(rank_map[pred])

    for prev, nxt in zip(ranks, ranks[1:]):
        assert nxt >= prev, f"predicted rank regressed: {ranks}"


def test_off_by_2_plus_rate_is_low_on_clean_data():
    """On the toy dataset the model should rarely make catastrophic ordinal errors."""
    df = _toy_dataset(n_per_class=50)
    _, report = train_test_eval(df, random_state=0)

    assert report.off_by_2_plus_rate < 0.10, (
        f"off-by->=2 rate {report.off_by_2_plus_rate:.3f} should be near zero on clean data"
    )


def test_handles_degenerate_threshold_without_crashing():
    """If a class is missing entirely, fit must not crash on a degenerate binary split."""
    df = pd.DataFrame(
        [
            {"text": "alpha alpha", "urgency": "low"},
            {"text": "alpha beta", "urgency": "low"},
            {"text": "gamma delta", "urgency": "high"},
            {"text": "gamma epsilon", "urgency": "high"},
        ]
        * 5
    )
    # 'medium' and 'critical' are absent — two of the three binary thresholds
    # collapse to constants.
    pipe = build_pipeline()
    pipe.fit(df["text"].tolist(), df["urgency"].tolist())

    preds = pipe.predict(df["text"].tolist())
    assert set(preds).issubset({"low", "high"})


def test_evaluate_returns_zero_mae_for_perfect_predictor():
    """Sanity: evaluate's MAE math is correct when prediction == truth."""
    df = _toy_dataset(n_per_class=20)

    class PerfectPipeline:
        def predict(self, X):
            # Cheat: we constructed _toy_dataset so text → label, but here we
            # just return the input's true label directly by lookup.
            mapping = dict(zip(df["text"], df["urgency"]))
            return np.array([mapping[x] for x in X])

    report = evaluate(PerfectPipeline(), df["text"].tolist(), df["urgency"].tolist())

    assert report.mae == 0.0
    assert report.off_by_2_plus_rate == 0.0
    assert report.macro_f1 == pytest.approx(1.0)


def test_ordinal_logreg_classes_attribute_is_set_after_fit():
    """sklearn convention: classes_ should be the fitted label set."""
    df = _toy_dataset(n_per_class=20)
    clf = OrdinalLogReg()
    # Fit directly on a tiny dense matrix to bypass the TF-IDF step.
    X = np.array([[len(text.split())] for text in df["text"]], dtype=float)
    clf.fit(X, df["urgency"].tolist())

    assert list(clf.classes_) == URGENCY_ORDER
