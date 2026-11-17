"""Model candidates, hyperparameter grids and evaluation helpers.

Kept separate from ``train.py`` so the notebook and the training script can
import the exact same definitions instead of re-typing them.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


# --------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------


def get_pipelines() -> dict[str, Pipeline]:
    """Return one pipeline per candidate model.

    Preprocessing lives inside the pipeline so that it is refitted inside every
    cross-validation fold (no information leaks from the validation folds).
    """
    return {
        "logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hgb": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        early_stopping=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def get_grids() -> dict[str, dict]:
    """Hyperparameter grids searched with ``GridSearchCV``.

    Deliberately small and exhaustive rather than random, so the search is
    deterministic and easy to report.
    """
    return {
        "logreg": {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__solver": ["liblinear", "lbfgs"],
        },
        "rf": {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 8, 15],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features": ["sqrt", 0.5],
            "clf__class_weight": ["balanced", "balanced_subsample"],
        },
        "hgb": {
            "clf__max_iter": [200, 400],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_leaf_nodes": [15, 31, 63],
            "clf__min_samples_leaf": [10, 20, 40],
            "clf__l2_regularization": [0.0, 1.0],
        },
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def f2_score(y_true, y_pred) -> float:
    """F-beta with beta=2: recall counts four times as much as precision."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denominator = 4 * precision + recall
    return (5 * precision * recall / denominator) if denominator else 0.0


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Score a probability vector at a given decision threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denominator = 4 * precision + recall
    f2 = (5 * precision * recall / denominator) if denominator else 0.0

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def tune_threshold(y_true, oof_prob, thresholds=None) -> float:
    """Pick the decision threshold maximising F2 on out-of-fold predictions.

    Out-of-fold predictions are used so the threshold is not tuned on data the
    model was fitted on.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)
    scores = [f2_score(y_true, (oof_prob >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])
