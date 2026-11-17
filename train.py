"""Train the laboratory quality prediction model.

This script is the notebook's model-training logic exported to a standalone,
command-line tool. It mirrors notebook cells "Model selection & hyperparameter
tuning" (sections 8-9) step for step.

Usage:
    python train.py                       # full grid search
    python train.py --fast                # skip grid search (smoke test)
    python train.py --data data/raw/ai4i2020.csv --out-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from src.data import (
    DEFAULT_DATA_PATH,
    FEATURE_NAMES,
    TARGET_COL,
    TARGET_DEFINITION,
    build_features,
    load_data,
    make_target,
)
from src.modeling import (
    SEED,
    evaluate,
    get_grids,
    get_pipelines,
    tune_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the laboratory quality prediction model (Pass/Fail)."
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA_PATH), help="path to the raw AI4I CSV"
    )
    parser.add_argument(
        "--out-dir", default="artifacts", help="directory for model.joblib and metrics.json"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the hyperparameter grid search (use default hyperparameters)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of downloading the dataset when it is missing",
    )
    return parser.parse_args()


def tune_models(
    X: pd.DataFrame, y: pd.Series, folds: int, seed: int, fast: bool
) -> pd.DataFrame:
    """Grid-search every candidate model; return one row per model."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pipelines = get_pipelines()
    grids = get_grids()

    rows = []
    for name, pipeline in pipelines.items():
        print(f"\n--- Tuning '{name}' ---")
        if fast:
            best_params: dict = {}
            cv_score = float(
                cross_val_predict_score(pipeline, X, y, cv)
            )
        else:
            search = GridSearchCV(
                pipeline,
                grids[name],
                scoring="average_precision",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y)
            best_params = {k: v for k, v in search.best_params_.items()}
            cv_score = float(search.best_score_)
            print(f"best params : {best_params}")
        print(f"CV PR-AUC   : {cv_score:.4f}")
        rows.append(
            {
                "model": name,
                "cv_pr_auc": cv_score,
                "best_params": best_params,
                "pipeline": pipelines[name],
            }
        )
    return pd.DataFrame(rows)


def cross_val_predict_score(pipeline, X, y, cv) -> float:
    """Mean PR-AUC across folds for the ``--fast`` path (no grid search)."""
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(y.iloc[val_idx], prob))
    return float(np.mean(scores))


def feature_importance(
    pipeline, X_test: pd.DataFrame, y_test: pd.Series, seed: int
) -> pd.DataFrame:
    """Permutation importance on the test set (works for any estimator)."""
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=seed,
        scoring="average_precision",
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X_test.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def print_results(results: pd.DataFrame, metrics: dict, threshold: float) -> None:
    print("\n=== Model comparison (5-fold CV on train, metric: PR-AUC) ===")
    header = f"{'model':<10}{'cv_pr_auc':>12}{'roc_auc':>10}{'f2':>8}{'recall':>9}"
    print(header)
    print("-" * len(header))
    for _, row in results.iterrows():
        m = row["eval"]
        print(
            f"{row['model']:<10}{row['cv_pr_auc']:>12.4f}"
            f"{m['roc_auc']:>10.4f}{m['f2']:>8.4f}{m['recall']:>9.4f}"
        )

    print(f"\n=== Final model: {results.iloc[0]['model']}  (threshold={threshold:.2f}) ===")
    print(f"Test PR-AUC  : {metrics['pr_auc']:.4f}")
    print(f"Test ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"Recall       : {metrics['recall']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f}")
    print(f"F2           : {metrics['f2']:.4f}")
    print(f"Accuracy     : {metrics['accuracy']:.4f}  (misleading at 3.4% prevalence)")
    print(f"Confusion    : {metrics['confusion_matrix']}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    df = load_data(args.data, allow_download=not args.no_download)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {args.data}")

    y = make_target(df)  # quality_fail: 1 = FAIL quality inspection
    X = build_features(df)  # inputs only
    assert list(X.columns) == FEATURE_NAMES
    print(f"Target '{TARGET_COL}' (1 = FAIL): {int(y.sum()):,} failures "
          f"({y.mean() * 100:.2f}% of rows)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    print(f"Train: {len(X_train):,} rows ({int(y_train.sum())} FAIL) | "
          f"Test: {len(X_test):,} rows ({int(y_test.sum())} FAIL)")

    # ------------------------------------------------------------------
    # 2. Model selection and hyperparameter tuning
    # ------------------------------------------------------------------
    results = tune_models(X_train, y_train, args.cv_folds, args.seed, args.fast)

    # Score each candidate on the held-out test set at its own best threshold
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    evals: list[dict] = []
    thresholds: list[float] = []
    for _, row in results.iterrows():
        pipeline = row["pipeline"]
        if row["best_params"]:
            pipeline.set_params(**row["best_params"])
        pipeline.fit(X_train, y_train)

        oof = cross_val_predict(
            pipeline, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        threshold = tune_threshold(y_train, oof)
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        thresholds.append(threshold)
        evals.append(evaluate(y_test, test_prob, threshold))
    results["threshold"] = thresholds
    results["eval"] = evals

    # The winner is the model with the best CV PR-AUC.
    results = results.sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    best = results.iloc[0]
    best_name = best["model"]
    best_pipeline = best["pipeline"]
    best_threshold = float(best["threshold"])
    print(f"\nSelected model: {best_name}")

    # ------------------------------------------------------------------
    # 3. Refit the winner on the full training split
    # ------------------------------------------------------------------
    best_pipeline.fit(X_train, y_train)
    test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_prob, best_threshold)

    importance = feature_importance(best_pipeline, X_test, y_test, args.seed)
    print("\nTop 5 features by permutation importance:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']:<16}{row['importance_mean']:.5f}")

    # ------------------------------------------------------------------
    # 4. Save artifacts
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": best_pipeline,
        "threshold": best_threshold,
        "feature_names": list(FEATURE_NAMES),
        "metrics": metrics,
        "results": results.drop(columns=["pipeline"]).to_dict(orient="records"),
        "importance": importance.to_dict(orient="records"),
        "target_definition": TARGET_DEFINITION,
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, out_dir / "model.joblib")

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": best_name,
                "threshold": best_threshold,
                "target_definition": TARGET_DEFINITION,
                "test_metrics": metrics,
                "cv_results": results.drop(columns=["pipeline"]).to_dict(orient="records"),
                "feature_importance": importance.to_dict(orient="records"),
                "trained_at": artifact["trained_at"],
                "sklearn_version": artifact["sklearn_version"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print_results(results, metrics, best_threshold)
    print(f"\nSaved model     -> {out_dir / 'model.joblib'}")
    print(f"Saved metrics   -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
