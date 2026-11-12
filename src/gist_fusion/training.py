from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gist_fusion.config import data_path, seed_everything
from gist_fusion.data import load_features, split_by_center
from gist_fusion.metrics import (
    PROBABILITY_COLUMNS,
    auc_values,
    bootstrap_estimates,
    classification_values,
    confusion_rows,
)
from gist_fusion.models import fit_final_model, save_artifacts


def retrain_final_models(config: dict, output_dir: str | Path, replicates: int = 0) -> dict:
    """Retrain the four final classifiers from their frozen, deidentified model inputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = config["experiment"]
    seed_everything(experiment["seed"])
    all_features = sorted({
        feature
        for columns in config["feature_sets"].values()
        for feature in columns
    })
    features = load_features(data_path(config, config["paths"]["features"]), all_features)
    train, validation = split_by_center(
        features,
        experiment["train_centers"],
        experiment["validation_centers"],
    )
    summary = {
        "status": "success",
        "mode": "retraining from frozen model inputs",
        "training_n": len(train),
        "validation_n": len(validation),
        "models": {},
    }
    all_confusion = []
    for model_name, feature_columns in config["feature_sets"].items():
        specification = config["models"][model_name]
        model, scaler, train_pred, validation_pred = fit_final_model(
            train,
            validation,
            feature_columns,
            specification["type"],
            specification["params"],
            experiment["seed"],
        )
        model_dir = output_dir / model_name.lower().replace("-", "_")
        save_artifacts(model_dir, model_name, model, scaler, feature_columns)
        train_pred.to_csv(model_dir / "training_predictions.csv", index=False)
        validation_pred.to_csv(model_dir / "validation_predictions.csv", index=False)

        labels = validation["true_label"].to_numpy(int)
        probabilities = validation_pred[PROBABILITY_COLUMNS].to_numpy(float)
        point = auc_values(labels, probabilities)
        if replicates:
            point, intervals, _ = bootstrap_estimates(
                labels, probabilities, auc_values, replicates=replicates, seed=experiment["seed"]
            )
        else:
            intervals = {}
        metrics = {
            **classification_values(labels, validation_pred["pred_label"].to_numpy(int)),
            **point,
        }
        if intervals:
            metrics["confidence_intervals"] = {key: list(value) for key, value in intervals.items()}
        summary["models"][model_name] = {
            "features": feature_columns,
            "validation_metrics": metrics,
        }
        all_confusion.extend(confusion_rows(
            model_name, labels, validation_pred["pred_label"].to_numpy(int)
        ))
    pd.DataFrame(all_confusion).to_csv(output_dir / "validation_confusion_matrices.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
