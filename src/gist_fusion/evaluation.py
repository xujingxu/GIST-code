from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gist_fusion.config import data_path
from gist_fusion.data import load_predictions, select_validation
from gist_fusion.metrics import (
    CLASS_NAMES,
    PROBABILITY_COLUMNS,
    auc_values,
    bootstrap_estimates,
    confusion_rows,
    continuous_nri_idi,
    paired_by_sample,
)

AUC_LABELS = {
    "class_0_auc": "Low-risk AUC",
    "class_1_auc": "Intermediate-risk AUC",
    "class_2_auc": "High-risk AUC",
    "macro_auc": "Macro-average AUC",
    "weighted_auc": "Weighted-average AUC",
    "micro_auc": "Micro-average AUC",
}


def reproduce_paper_metrics(config: dict, output_dir: str | Path, replicates: int | None = None) -> dict:
    """Recompute the reported validation metrics from frozen anonymized predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = config["experiment"]
    if replicates is None:
        replicates = experiment["bootstrap_replicates"]
    all_predictions = {}
    auc_rows, confusion = [], []

    allowed_centers = sorted(
        set(experiment["train_centers"]) | set(experiment["validation_centers"])
    )
    for model_name, relative_path in config["paths"]["predictions"].items():
        frame = load_predictions(data_path(config, relative_path))
        validation = select_validation(frame, experiment["validation_centers"], allowed_centers)
        all_predictions[model_name] = validation
        labels = validation["true_label"].to_numpy(int)
        probabilities = validation[PROBABILITY_COLUMNS].to_numpy(float)
        point, intervals, effective = bootstrap_estimates(
            labels, probabilities, auc_values, replicates=replicates, seed=experiment["seed"]
        )
        for key, label in AUC_LABELS.items():
            lower, upper = intervals[key]
            auc_rows.append({
                "model": model_name,
                "metric": label,
                "estimate": point[key],
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_replicates": effective,
            })
        confusion.extend(confusion_rows(model_name, labels, validation["pred_label"].to_numpy(int)))

    paired = paired_by_sample(all_predictions["Fusion"], all_predictions["Radiomics-Clinical"])
    labels = paired["true_label_new"].to_numpy(int)
    old = paired[[f"{column}_old" for column in PROBABILITY_COLUMNS]].to_numpy(float)
    new = paired[[f"{column}_new" for column in PROBABILITY_COLUMNS]].to_numpy(float)

    # The estimator needs resampled old/new probabilities, so perform paired bootstrap explicitly.
    point = continuous_nri_idi(labels, old, new)
    rng = np.random.default_rng(experiment["seed"] + 1)
    samples = {key: [] for key in point}
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), len(labels))
        values = continuous_nri_idi(labels[indices], old[indices], new[indices])
        for key, value in values.items():
            if np.isfinite(value):
                samples[key].append(value)

    nri_rows = []
    for k, class_name in enumerate(CLASS_NAMES):
        for measure in ("nri", "idi"):
            key = f"class_{k}_{measure}"
            lower, upper = np.percentile(samples[key], [2.5, 97.5])
            nri_rows.append({
                "scope": class_name,
                "measure": measure.upper(),
                "estimate": point[key],
                "ci_lower": lower,
                "ci_upper": upper,
            })
    for measure in ("nri", "idi"):
        key = f"macro_{measure}"
        lower, upper = np.percentile(samples[key], [2.5, 97.5])
        nri_rows.append({
            "scope": "Macro average",
            "measure": measure.upper(),
            "estimate": point[key],
            "ci_lower": lower,
            "ci_upper": upper,
        })

    pd.DataFrame(auc_rows).to_csv(output_dir / "validation_auc.csv", index=False)
    pd.DataFrame(confusion).to_csv(output_dir / "validation_confusion_matrices.csv", index=False)
    pd.DataFrame(nri_rows).to_csv(output_dir / "fusion_vs_radiomics_nri_idi.csv", index=False)
    summary = {
        "status": "success",
        "mode": "frozen-prediction evaluation",
        "validation_n": int(len(labels)),
        "class_counts": {CLASS_NAMES[k]: int((labels == k).sum()) for k in range(3)},
        "bootstrap_replicates": replicates,
        "seed_auc": experiment["seed"],
        "seed_nri_idi": experiment["seed"] + 1,
        "fusion_weighted_auc": next(
            row["estimate"] for row in auc_rows
            if row["model"] == "Fusion" and row["metric"] == "Weighted-average AUC"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
