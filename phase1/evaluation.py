from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from gmm_sampling import GMMSampling

CLASS_ORDER = ["H", "D", "A"]


def multiclass_brier(y_true, probabilities, classes=CLASS_ORDER) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(probabilities, dtype=float)
    for row, label in enumerate(y_true):
        onehot[row, class_to_idx[label]] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def ranked_probability_score(y_true, probabilities, classes=CLASS_ORDER) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(probabilities, dtype=float)
    for row, label in enumerate(y_true):
        onehot[row, class_to_idx[label]] = 1.0
    return float(
        np.mean(
            np.sum(
                (np.cumsum(probabilities, axis=1) - np.cumsum(onehot, axis=1)) ** 2,
                axis=1
            ) / (len(classes) - 1)
        )
    )


def geometric_mean_recall(y_true, y_pred, classes=CLASS_ORDER) -> float:
    _, recalls, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return float(gmean(np.clip(recalls, 1e-12, 1.0)))


def align_probabilities(estimator, X, classes=CLASS_ORDER):
    p = estimator.predict_proba(X)
    aligned = np.zeros((len(X), len(classes)), dtype=float)
    idx = {c: i for i, c in enumerate(estimator.classes_)}
    for j, c in enumerate(classes):
        if c in idx:
            aligned[:, j] = p[:, idx[c]]
    aligned = aligned / aligned.sum(axis=1, keepdims=True)
    return aligned


def evaluate_predictions(y_true, y_pred, probabilities, classes=CLASS_ORDER) -> dict:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([class_to_idx[v] for v in y_true])
    log_loss = -np.mean(
        np.log(
            np.clip(
                probabilities[np.arange(len(y_true)), y_true_idx],
                1e-15,
                1.0,
            )
        )
    )
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "g_mean": geometric_mean_recall(y_true, y_pred, classes),
        "log_loss": float(log_loss),
        "brier": multiclass_brier(y_true, probabilities, classes),
        "rps": ranked_probability_score(y_true, probabilities, classes),
    }


def run_raw_vs_gmm_experiment(
    pre_match: pd.DataFrame,
    feature_cols: list[str],
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, GMMSampling, dict]:
    train = pre_match.loc[pre_match.split.eq("train")].copy()
    test = pre_match.loc[pre_match.split.eq("test")].copy()
    if train.empty or test.empty:
        raise ValueError("Both train and test splits must be non-empty.")

    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train.outcome
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_test = test.outcome

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_test = X_test.fillna(medians).fillna(0.0)

    base_model = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=3,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )
    weighted_model = clone(base_model).set_params(class_weight="balanced")

    rows = []
    fitted = {}
    for name, model, X_fit, y_fit in [
        ("Vanilla", clone(base_model), X_train, y_train),
        ("ClassWeight", weighted_model, X_train, y_train),
    ]:
        start = time.perf_counter()
        model.fit(X_fit, y_fit)
        elapsed = time.perf_counter() - start
        pred = model.predict(X_test)
        prob = align_probabilities(model, X_test)
        rows.append(
            {
                "method": name,
                "train_seconds": elapsed,
                **evaluate_predictions(y_test, pred, prob)
            }
        )
        fitted[name] = model

    train_counts = y_train.value_counts()
    majority_label = train_counts.idxmax()
    minority_labels = tuple(c for c in CLASS_ORDER if c != majority_label)
    sampler = GMMSampling(
        minority_classes=minority_labels,
        majority_classes=(majority_label,),
        k_neighbors=5,
        max_components=6,
        random_state=random_state,
    )
    X_gmm, y_gmm = sampler.fit_resample(X_train, y_train)
    gmm_model = clone(base_model)
    start = time.perf_counter()
    gmm_model.fit(X_gmm, y_gmm)
    elapsed = time.perf_counter() - start
    pred = gmm_model.predict(X_test)
    prob = align_probabilities(gmm_model, X_test)
    rows.append(
        {
            "method": "GMMSampling",
            "train_seconds": elapsed,
            **evaluate_predictions(y_test, pred, prob)
        }
    )
    fitted["GMMSampling"] = gmm_model

    results = (
        pd.DataFrame(rows)
        .sort_values("balanced_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    extra = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_gmm": X_gmm,
        "y_gmm": y_gmm,
        "models": fitted,
    }
    return results, sampler, extra