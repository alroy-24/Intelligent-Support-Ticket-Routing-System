"""Ordinal logistic-regression baseline for urgency.

Why ordinal and not plain multi-class:
- Urgency is an ORDERED label: low < medium < high < critical.
- A plain multi-class model treats every misclassification equally — predicting
  "low" when the truth is "critical" is scored the same as predicting "high".
  That's wrong for routing: under-escalating a critical ticket has very different
  business cost from misclassifying it as high.
- An ordinal model is forced to learn the ordering, and we evaluate it with MAE
  on the integer ranks so distant errors hurt more.

Approach (Frank & Hall, 2001):
- For K ordered classes we train K-1 binary classifiers, each answering
  "is the label strictly greater than threshold k?". The shared TF-IDF features
  push all K-1 classifiers in the same direction for a given input, so the
  model implicitly learns the ordering.
- At inference we recover class probabilities from the cumulative ones:
    P(y=k) = P(y > k-1) - P(y > k)
  Threshold violations (non-monotonic cumulative probs) are possible — we clip
  and renormalise rather than enforce monotonicity at training time. Simpler
  and good enough for a baseline.

Why not CORAL / encoder approach in the baseline:
- Same reason TF-IDF + LogReg is the category baseline: it runs in seconds,
  gives a real number to beat, and surfaces whether the harder approach is
  earning its complexity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ticketrouting.schemas import Urgency

URGENCY_ORDER: list[str] = [
    Urgency.LOW.value,
    Urgency.MEDIUM.value,
    Urgency.HIGH.value,
    Urgency.CRITICAL.value,
]


class OrdinalLogReg(BaseEstimator, ClassifierMixin):
    """Frank & Hall (2001) ordinal classifier built from K-1 binary LogRegs.

    Each binary classifier answers P(y > k). Class probabilities are recovered
    from consecutive cumulative probabilities and clipped/renormalised.
    """

    def __init__(
        self,
        classes: list[str] | None = None,
        C: float = 1.0,
        class_weight: str | None = "balanced",
        max_iter: int = 1000,
    ):
        self.classes = classes if classes is not None else list(URGENCY_ORDER)
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    def _rank(self, y: list[str] | np.ndarray) -> np.ndarray:
        rank_map = {c: i for i, c in enumerate(self.classes)}
        return np.array([rank_map[v] for v in y])

    def fit(self, X, y):
        y_rank = self._rank(y)
        K = len(self.classes)
        self.classifiers_: list[LogisticRegression] = []
        for k in range(K - 1):
            y_bin = (y_rank > k).astype(int)
            # If a threshold is degenerate (all 0 or all 1) sklearn would refuse —
            # synthesise a fallback that always predicts the observed constant.
            if len(np.unique(y_bin)) < 2:
                const = int(y_bin[0])
                self.classifiers_.append(_ConstantBinary(const))
                continue
            clf = LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
            )
            clf.fit(X, y_bin)
            self.classifiers_.append(clf)
        self.classes_ = np.array(self.classes)
        return self

    def predict_proba(self, X) -> np.ndarray:
        K = len(self.classes)
        # cum_p[:, k] = P(y > k)
        cum_p = np.column_stack(
            [self._proba_positive(clf, X) for clf in self.classifiers_]
        )
        n = cum_p.shape[0]
        proba = np.zeros((n, K))
        proba[:, 0] = 1.0 - cum_p[:, 0]
        for k in range(1, K - 1):
            proba[:, k] = cum_p[:, k - 1] - cum_p[:, k]
        proba[:, K - 1] = cum_p[:, K - 2]
        # Threshold violations can produce small negatives; clip and renormalise.
        proba = np.clip(proba, 0.0, 1.0)
        row_sum = proba.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return proba / row_sum

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    @staticmethod
    def _proba_positive(clf, X) -> np.ndarray:
        if isinstance(clf, _ConstantBinary):
            return np.full(X.shape[0], float(clf.constant))
        return clf.predict_proba(X)[:, 1]


@dataclass
class _ConstantBinary:
    """Fallback for degenerate binary thresholds — always returns the same class."""

    constant: int  # 0 or 1


def build_pipeline(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    C: float = 1.0,
    classes: list[str] | None = None,
) -> Pipeline:
    """TF-IDF + ordinal LogReg pipeline. Same featurisation as the category baseline."""
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
            ("clf", OrdinalLogReg(C=C, classes=classes)),
        ]
    )


@dataclass
class UrgencyEvalReport:
    """Eval focused on ordinal-aware metrics; macro_f1 is reported for comparison only."""

    mae: float                          # mean |true_rank - pred_rank|, lower is better
    off_by_2_plus_rate: float           # share of predictions off by >=2 ranks
    macro_f1: float                     # for comparison with nominal classifiers
    weighted_f1: float
    per_class_f1: dict[str, float]
    confusion: np.ndarray
    labels: list[str]
    classification_report: str

    def pretty(self) -> str:
        lines = [
            f"MAE (rank):           {self.mae:.4f}",
            f"Off-by->=2 rate:      {self.off_by_2_plus_rate:.4f}",
            f"Macro F1:             {self.macro_f1:.4f}  (for cross-comparison only)",
            f"Weighted F1:          {self.weighted_f1:.4f}",
            "",
            "Per-class F1:",
        ]
        for label in self.labels:
            f1 = self.per_class_f1.get(label, float("nan"))
            lines.append(f"  {label:12s} {f1:.4f}")
        lines.append("")
        lines.append(self.classification_report)
        return "\n".join(lines)


def evaluate(
    pipe: Pipeline, X: list[str], y_true: list[str], labels: list[str] | None = None
) -> UrgencyEvalReport:
    labels = labels if labels is not None else list(URGENCY_ORDER)
    y_pred = pipe.predict(X)
    rank_map = {c: i for i, c in enumerate(labels)}
    true_rank = np.array([rank_map[v] for v in y_true])
    pred_rank = np.array([rank_map[v] for v in y_pred])
    abs_err = np.abs(true_rank - pred_rank)

    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return UrgencyEvalReport(
        mae=float(abs_err.mean()),
        off_by_2_plus_rate=float((abs_err >= 2).mean()),
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
    label_col: str = "urgency",
    test_size: float = 0.2,
    random_state: int = 42,
    pipeline: Pipeline | None = None,
) -> tuple[Pipeline, UrgencyEvalReport]:
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


def predict_one(pipe: Pipeline, text: str) -> tuple[Urgency, float]:
    """Predict urgency for a single ticket and return (urgency, confidence)."""
    proba = pipe.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    label = pipe.named_steps["clf"].classes_[idx]
    return Urgency(label), float(proba[idx])


def save(pipe: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
