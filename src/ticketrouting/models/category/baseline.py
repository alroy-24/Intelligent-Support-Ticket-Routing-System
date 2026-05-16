"""TF-IDF + Logistic Regression baseline for category classification.

Why a baseline:
- Trains in seconds, runs in microseconds. Sets a real number to beat before we
  spend time on transformers.
- If DistilBERT only beats it by a couple of macro-F1 points, that's a real
  signal about the cost/benefit of the bigger model — not a victory.

Scope note:
- This is **multi-class** (one category per ticket), not multi-label. Bitext has
  one intent per row so we can't honestly train multi-label here. The README's
  multi-label framing is the production target; the baseline is the honest
  starting point given the data we have.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ticketrouting.schemas import Category


def build_pipeline(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    C: float = 1.0,
) -> Pipeline:
    """Build the TF-IDF + LogReg pipeline.

    Defaults are tuned for short customer-support text:
    - bigrams help with phrases like "log in", "credit card"
    - min_df=2 drops single-occurrence tokens (mostly typos)
    - balanced class weights because Bitext is skewed toward billing
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


@dataclass
class EvalReport:
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    confusion: np.ndarray
    labels: list[str]
    classification_report: str

    def pretty(self) -> str:
        lines = [
            f"Macro F1:    {self.macro_f1:.4f}",
            f"Weighted F1: {self.weighted_f1:.4f}",
            "",
            "Per-class F1:",
        ]
        for label, f1 in sorted(self.per_class_f1.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {label:20s} {f1:.4f}")
        lines.append("")
        lines.append(self.classification_report)
        return "\n".join(lines)


def evaluate(pipe: Pipeline, X: list[str], y_true: list[str]) -> EvalReport:
    y_pred = pipe.predict(X)
    labels = sorted(set(y_true) | set(y_pred))

    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return EvalReport(
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        per_class_f1={label: float(f1) for label, f1 in zip(labels, per_class)},
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        labels=labels,
        classification_report=classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    )


def train_test_eval(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "category",
    test_size: float = 0.2,
    random_state: int = 42,
    pipeline: Pipeline | None = None,
) -> tuple[Pipeline, EvalReport]:
    """Stratified train/test split, fit, evaluate. Returns the fitted pipeline + report."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].tolist(),
        df[label_col].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    pipe = pipeline or build_pipeline()
    pipe.fit(X_train, y_train)
    return pipe, evaluate(pipe, X_test, y_test)


def predict_one(pipe: Pipeline, text: str) -> tuple[Category, float]:
    """Predict the category for a single ticket and return (category, confidence)."""
    proba = pipe.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    label = pipe.classes_[idx]
    return Category(label), float(proba[idx])


def save(pipe: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
