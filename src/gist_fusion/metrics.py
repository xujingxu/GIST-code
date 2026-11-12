from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ("Low risk", "Intermediate risk", "High risk")
PROBABILITY_COLUMNS = ["pred_prob_0", "pred_prob_1", "pred_prob_2"]


def auc_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float] | None:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    binary = np.column_stack([labels == k for k in range(3)]).astype(int)
    support = binary.sum(axis=0)
    if np.any(support == 0) or np.any(support == len(labels)):
        return None
    class_aucs = [roc_auc_score(binary[:, k], probabilities[:, k]) for k in range(3)]
    return {
        **{f"class_{k}_auc": class_aucs[k] for k in range(3)},
        "macro_auc": float(np.mean(class_aucs)),
        "weighted_auc": float(np.average(class_aucs, weights=support)),
        "micro_auc": roc_auc_score(binary.ravel(), probabilities.ravel()),
    }


def classification_values(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "weighted_sensitivity": recall_score(labels, predictions, average="weighted"),
        "macro_sensitivity": recall_score(labels, predictions, average="macro"),
        "weighted_precision": precision_score(labels, predictions, average="weighted"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def bootstrap_estimates(
    labels: np.ndarray,
    probabilities: np.ndarray,
    estimator: Callable[[np.ndarray, np.ndarray], dict[str, float] | None],
    replicates: int = 2000,
    seed: int = 42,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    point = estimator(labels, probabilities)
    if point is None:
        raise ValueError("All three classes must be present")
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in point}
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), len(labels))
        values = estimator(labels[indices], probabilities[indices])
        if values is not None:
            for key, value in values.items():
                if np.isfinite(value):
                    samples[key].append(value)
    intervals = {
        key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for key, values in samples.items()
    }
    return point, intervals, min(map(len, samples.values()))


def continuous_nri_idi(
    labels: np.ndarray,
    old_probabilities: np.ndarray,
    new_probabilities: np.ndarray,
) -> dict[str, float]:
    """Class-by-class one-vs-rest continuous NRI and discrimination-slope IDI."""
    labels = np.asarray(labels, dtype=int)
    old = np.asarray(old_probabilities, dtype=float)
    new = np.asarray(new_probabilities, dtype=float)
    result: dict[str, float] = {
        f"class_{k}_{measure}": np.nan
        for k in range(3)
        for measure in ("nri", "idi")
    }
    result["macro_nri"] = np.nan
    result["macro_idi"] = np.nan
    nris, idis = [], []
    for k in range(3):
        event = labels == k
        nonevent = ~event
        if not event.any() or not nonevent.any():
            continue
        delta = new[:, k] - old[:, k]
        up, down = delta > 0, delta < 0
        nri = (
            up[event].mean() - down[event].mean()
            + down[nonevent].mean() - up[nonevent].mean()
        )
        idi = (
            new[event, k].mean() - old[event, k].mean()
            - new[nonevent, k].mean() + old[nonevent, k].mean()
        )
        result[f"class_{k}_nri"] = float(nri)
        result[f"class_{k}_idi"] = float(idi)
        nris.append(nri)
        idis.append(idi)
    if nris:
        result["macro_nri"] = float(np.mean(nris))
        result["macro_idi"] = float(np.mean(idis))
    return result


def confusion_rows(model: str, labels: np.ndarray, predictions: np.ndarray) -> list[dict]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    rows = []
    for k, class_name in enumerate(CLASS_NAMES):
        rows.append({
            "model": model,
            "true_class": class_name,
            "predicted_low": int(matrix[k, 0]),
            "predicted_intermediate": int(matrix[k, 1]),
            "predicted_high": int(matrix[k, 2]),
        })
    return rows


def paired_by_sample(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    merged = first.merge(second, on="sample_id", suffixes=("_new", "_old"), validate="one_to_one")
    if len(merged) != len(first) or len(merged) != len(second):
        raise ValueError("Prediction tables do not contain the same patients")
    if not (merged["true_label_new"] == merged["true_label_old"]).all():
        raise ValueError("True labels differ between paired prediction tables")
    return merged
