"""Smoke tests for the baseline classifier on a tiny synthetic dataset."""
from __future__ import annotations

import pandas as pd
import pytest

from ticketrouting.models.category.baseline import (
    build_pipeline,
    evaluate,
    load,
    predict_one,
    save,
    train_test_eval,
)
from ticketrouting.schemas import Category


def _toy_dataset() -> pd.DataFrame:
    """Mock support tickets — clearly separable so the classifier should nail them."""
    rows = []
    for _ in range(15):
        rows += [
            ("My credit card was charged twice for the same order", Category.BILLING.value),
            ("I want a refund for my last invoice", Category.BILLING.value),
            ("Payment failed when I tried to renew my subscription", Category.BILLING.value),
            ("I can't log in, my password reset link expired", Category.ACCOUNT.value),
            ("Please delete my account and remove my email from your system", Category.ACCOUNT.value),
            ("How do I change my username and profile photo?", Category.ACCOUNT.value),
            ("The application crashes whenever I open the reports page", Category.BUG.value),
            ("Search results show duplicate entries that shouldn't exist", Category.BUG.value),
        ]
    return pd.DataFrame(rows, columns=["text", "category"])


def test_pipeline_fits_and_predicts_on_toy_data():
    df = _toy_dataset()
    pipe, report = train_test_eval(df, test_size=0.25, random_state=0)

    # On clearly-separable toy data the baseline should be effectively perfect.
    assert report.macro_f1 >= 0.9, f"unexpectedly weak: {report.macro_f1:.3f}"


def test_predict_one_returns_category_and_confidence():
    df = _toy_dataset()
    pipe, _ = train_test_eval(df, test_size=0.25, random_state=0)

    cat, conf = predict_one(pipe, "I was charged twice on my credit card")
    assert cat == Category.BILLING
    assert 0.0 <= conf <= 1.0
    assert conf > 0.3  # baseline shouldn't be uncertain on an in-distribution billing ticket


def test_save_and_load_roundtrip(tmp_path):
    df = _toy_dataset()
    pipe, _ = train_test_eval(df, test_size=0.25, random_state=0)

    path = tmp_path / "model.joblib"
    save(pipe, path)
    loaded = load(path)

    cat_before, _ = predict_one(pipe, "please reset my password")
    cat_after, _ = predict_one(loaded, "please reset my password")
    assert cat_before == cat_after


def test_build_pipeline_returns_fittable_sklearn_pipeline():
    pipe = build_pipeline(max_features=100)
    df = _toy_dataset()
    pipe.fit(df["text"], df["category"])
    report = evaluate(pipe, df["text"].tolist(), df["category"].tolist())
    assert report.macro_f1 > 0
