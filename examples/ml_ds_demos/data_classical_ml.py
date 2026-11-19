"""Data contracts, leakage-safe classical ML, clustering, and core math.

Run after installing examples/requirements-ml-ds.txt:
    python examples/ml_ds_demos/data_classical_ml.py

The dataset is synthetic and deterministic. The point is the engineering loop:
validate data, split by entity, fit preprocessing only on training data, choose
a threshold for a stated objective, inspect several model families, and retain
enough evidence to reproduce the result.
"""

# %% Imports
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


FEATURE_COLUMNS = ["duration_ms", "error_count", "region"]
NUMERIC_COLUMNS = ["duration_ms", "error_count"]
CATEGORICAL_COLUMNS = ["region"]


@dataclass(frozen=True)
class ClassificationMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    threshold: float


# %% Step 1 — ingestion/data contract before model code
def make_demo_events(seed: int = 17, rows: int = 360) -> pd.DataFrame:
    """Generate a small grouped event dataset with imbalance and missingness."""
    if rows % 4:
        raise ValueError("rows must be divisible by 4 so every customer has four events")
    rng = np.random.default_rng(seed)
    customer_id = np.repeat(np.arange(rows // 4), 4)
    region = rng.choice(["us", "eu", "apac"], size=rows, p=[0.50, 0.30, 0.20])
    duration_ms = rng.lognormal(mean=5.1, sigma=0.55, size=rows)
    error_count = rng.poisson(lam=1.0, size=rows)
    # This target is only for a teaching demo; production labels need provenance and QA.
    log_odds = -3.0 + 0.014 * duration_ms + 0.65 * error_count + 0.45 * (region == "apac")
    incident = (rng.random(rows) < 1 / (1 + np.exp(-log_odds))).astype(int)
    frame = pd.DataFrame(
        {
            "event_id": [f"evt-{index:04d}" for index in range(rows)],
            "customer_id": customer_id,
            "event_time": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "duration_ms": duration_ms,
            "error_count": error_count,
            "region": region,
            "incident": incident,
        }
    )
    missing_rows = rng.choice(frame.index, size=18, replace=False)
    frame.loc[missing_rows, "duration_ms"] = np.nan
    return frame


def validate_event_contract(frame: pd.DataFrame) -> list[str]:
    """Return violations; missing numeric values are allowed and imputed later."""
    required = {"event_id", "customer_id", "event_time", *FEATURE_COLUMNS, "incident"}
    errors: list[str] = []
    missing = sorted(required.difference(frame.columns))
    if missing:
        errors.append(f"missing columns: {missing}")
    if not missing:
        if frame["event_id"].duplicated().any():
            errors.append("event_id must be unique")
        if (frame["duration_ms"].dropna() < 0).any():
            errors.append("duration_ms cannot be negative")
        if not set(frame["incident"].dropna().unique()).issubset({0, 1}):
            errors.append("incident must be binary")
        if frame["customer_id"].isna().any():
            errors.append("customer_id is required for leakage-safe splitting")
    return errors


# %% Step 2 — split by customer before fitting transformations
def split_by_customer(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    train_indices, test_indices = next(splitter.split(frame, frame["incident"], groups=frame["customer_id"]))
    train, test = frame.iloc[train_indices].copy(), frame.iloc[test_indices].copy()
    if set(train["customer_id"]).intersection(test["customer_id"]):
        raise AssertionError("customer leakage across train/test split")
    return train, test


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def threshold_for_minimum_recall(labels: np.ndarray, probabilities: np.ndarray, minimum_recall: float = 0.80) -> float:
    """Pick the highest-precision threshold satisfying a business recall floor."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    candidates = [
        (precision[index], thresholds[index])
        for index in range(len(thresholds))
        if recall[index] >= minimum_recall
    ]
    return float(max(candidates, default=(0.0, 0.5))[1])


def score_classifier(labels: np.ndarray, probabilities: np.ndarray) -> ClassificationMetrics:
    threshold = threshold_for_minimum_recall(labels, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return ClassificationMetrics(
        pr_auc=float(average_precision_score(labels, probabilities)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        brier_score=float(brier_score_loss(labels, probabilities)),
        threshold=threshold,
    )


# %% Step 3 — compare interpretable, ensemble, and gradient-boosting baselines
def train_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, ClassificationMetrics]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=7),
        "random_forest": RandomForestClassifier(
            n_estimators=120, max_depth=6, class_weight="balanced", random_state=7, n_jobs=1
        ),
        "xgboost": XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            eval_metric="logloss",
            random_state=7,
            n_jobs=1,
        ),
    }
    results: dict[str, ClassificationMetrics] = {}
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipeline.fit(train[FEATURE_COLUMNS], train["incident"])
        probabilities = pipeline.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        results[name] = score_classifier(test["incident"].to_numpy(), probabilities)
    return results


# %% Step 4 — unsupervised clustering and math checkpoints
def cluster_events(frame: pd.DataFrame) -> dict[str, int]:
    numeric = frame[NUMERIC_COLUMNS].copy()
    numeric["duration_ms"] = numeric["duration_ms"].fillna(numeric["duration_ms"].median())
    matrix = StandardScaler().fit_transform(numeric)
    kmeans_labels = KMeans(n_clusters=3, n_init=10, random_state=7).fit_predict(matrix)
    dbscan_labels = DBSCAN(eps=0.55, min_samples=5).fit_predict(matrix)
    return {
        "kmeans_clusters": int(len(set(kmeans_labels))),
        "dbscan_clusters_excluding_noise": int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
        "dbscan_noise_points": int(np.sum(dbscan_labels == -1)),
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(left, right) / denominator)


def gradient_descent_linear_regression() -> tuple[float, float]:
    """Show calculus/optimization mechanics with a deterministic linear problem."""
    features = np.linspace(-1, 1, 30)
    targets = 2.0 * features + 0.5
    weight, bias, learning_rate = 0.0, 0.0, 0.2
    initial_loss = float(np.mean((weight * features + bias - targets) ** 2))
    for _ in range(80):
        residual = weight * features + bias - targets
        weight -= learning_rate * float(2 * np.mean(residual * features))
        bias -= learning_rate * float(2 * np.mean(residual))
    final_loss = float(np.mean((weight * features + bias - targets) ** 2))
    return initial_loss, final_loss


def demo() -> None:
    events = make_demo_events()
    errors = validate_event_contract(events)
    if errors:
        raise ValueError(f"data contract failed: {errors}")
    train, test = split_by_customer(events)
    metrics = train_baselines(train, test)
    clusters = cluster_events(events)
    initial_loss, final_loss = gradient_descent_linear_regression()
    cosine = cosine_similarity(np.array([1.0, 2.0]), np.array([2.0, 4.0]))

    print("Data contract: PASS; train/test customers:", train["customer_id"].nunique(), test["customer_id"].nunique())
    print("Classical ML metrics:", {name: asdict(score) for name, score in metrics.items()})
    print("Clustering summary:", clusters)
    print("Gradient-descent loss:", {"initial": initial_loss, "final": final_loss})
    print("Cosine([1, 2], [2, 4]):", cosine)

    assert not set(train["customer_id"]).intersection(test["customer_id"])
    assert all(0 <= score.pr_auc <= 1 for score in metrics.values())
    assert final_loss < initial_loss
    assert np.isclose(cosine, 1.0)
    print("PASS: data contract, leakage-safe ML, clustering, and math checks completed.")


if __name__ == "__main__":
    demo()
