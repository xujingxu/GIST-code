from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from gist_fusion.metrics import PROBABILITY_COLUMNS


def build_classifier(model_type: str, params: dict[str, Any], seed: int):
    params = dict(params)
    if model_type == "xgboost":
        return XGBClassifier(**params, random_state=seed)
    if model_type == "svm":
        return SVC(**params, random_state=seed)
    raise ValueError(f"Unsupported classifier type: {model_type}")


def predict_frame(
    model,
    scaler: StandardScaler,
    samples: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    features = samples[feature_columns].to_numpy(float)
    scaled = scaler.transform(features)
    probabilities = model.predict_proba(scaled)
    predictions = model.predict(scaled).astype(int)
    output = samples[["sample_id", "center", "true_label"]].copy().reset_index(drop=True)
    output["pred_label"] = predictions
    for k, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = probabilities[:, k]
    return output


def fit_final_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
    params: dict[str, Any],
    seed: int,
) -> tuple[Any, StandardScaler, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[feature_columns].to_numpy(float))
    model = build_classifier(model_type, params, seed)
    model.fit(train_scaled, train["true_label"].to_numpy(int))
    train_predictions = predict_frame(model, scaler, train, feature_columns)
    validation_predictions = predict_frame(model, scaler, validation, feature_columns)
    return model, scaler, train_predictions, validation_predictions


def save_artifacts(
    output_dir: str | Path,
    model_name: str,
    model,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace("-", "_")
    joblib.dump(model, output_dir / f"{safe_name}_model.joblib")
    joblib.dump(scaler, output_dir / f"{safe_name}_scaler.joblib")
    (output_dir / f"{safe_name}_features.txt").write_text(
        "\n".join(feature_columns) + "\n", encoding="utf-8"
    )
