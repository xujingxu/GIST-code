from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_PREDICTION_COLUMNS = [
    "sample_id",
    "center",
    "true_label",
    "pred_label",
    "pred_prob_0",
    "pred_prob_1",
    "pred_prob_2",
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return pd.read_csv(path, dtype={"sample_id": "string"})


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique within a prediction table")
    labels = set(frame["true_label"].unique())
    if not labels.issubset({0, 1, 2}):
        raise ValueError(f"Unexpected true labels: {sorted(labels)}")
    predictions = set(frame["pred_label"].unique())
    if not predictions.issubset({0, 1, 2}):
        raise ValueError(f"Unexpected predicted labels: {sorted(predictions)}")
    probabilities = frame[["pred_prob_0", "pred_prob_1", "pred_prob_2"]].to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction probabilities contain missing or infinite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Prediction probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each patient's class probabilities must sum to 1")


def load_predictions(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path)
    validate_prediction_frame(frame)
    return frame


def load_features(path: str | Path, required_columns: list[str]) -> pd.DataFrame:
    frame = _read_csv(path)
    required = {"sample_id", "center", "true_label", *required_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique in the feature table")
    numeric = frame[list(required_columns)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Model features contain missing or infinite values")
    return frame


def split_by_center(
    frame: pd.DataFrame,
    train_centers: list[int],
    validation_centers: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap = set(train_centers) & set(validation_centers)
    if overlap:
        raise ValueError(f"Training and validation centers overlap: {sorted(overlap)}")
    train = frame[frame["center"].isin(train_centers)].copy()
    validation = frame[frame["center"].isin(validation_centers)].copy()
    if train.empty or validation.empty:
        raise ValueError("Both training and validation cohorts must contain patients")
    known = set(train_centers) | set(validation_centers)
    unknown = set(frame["center"].unique()) - known
    if unknown:
        raise ValueError(f"Rows with unassigned centers: {sorted(unknown)}")
    return train, validation


def select_validation(
    frame: pd.DataFrame,
    validation_centers: list[int],
    allowed_centers: list[int] | None = None,
) -> pd.DataFrame:
    """Select validation rows while accepting either full-cohort or validation-only tables."""
    validation = frame[frame["center"].isin(validation_centers)].copy()
    if validation.empty:
        raise ValueError("The validation cohort contains no patients")
    known = set(allowed_centers or validation_centers)
    unknown = set(frame["center"].unique()) - known
    if unknown:
        raise ValueError(f"Rows with unassigned centers: {sorted(unknown)}")
    return validation
