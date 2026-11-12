#!/usr/bin/env python3
"""Build the repository's deidentified feature and prediction tables.

This helper is intended for the study authors, who already hold the authorized source
results. It never writes the Patient_ID-to-sample_id mapping, raw identifiers, source
paths, or any other linkage file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTION_OUTPUTS = {
    "fusion.csv": "fusion_final/baseline_predictions.csv",
    "radiomics_clinical.csv": "radiomics+clinical/baseline_predictions.csv",
    "habitat_clinical.csv": "habitat+clinical/baseline_predictions.csv",
    "ith_clinical.csv": "ITH+clinical/baseline_predictions.csv",
}


def _read_source(source_root: Path, relative_path: str) -> pd.DataFrame:
    path = source_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing authorized source result: {path}")
    frame = pd.read_csv(path)
    if "Patient_ID" not in frame.columns:
        raise ValueError(f"Source result has no Patient_ID column: {path.name}")
    return frame


def _standard_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({
        "Patient_ID": frame["Patient_ID"].astype(str),
        "center": frame["Center"].astype(int),
        "true_label": frame["True_Label"].astype(int),
        "pred_label": frame["Pred_Label"].astype(int),
        "pred_prob_0": frame["Pred_Prob_0"].astype(float),
        "pred_prob_1": frame["Pred_Prob_1"].astype(float),
        "pred_prob_2": frame["Pred_Prob_2"].astype(float),
    })
    _assert_valid_prediction(output)
    return output


def _assert_valid_prediction(frame: pd.DataFrame) -> None:
    if frame["Patient_ID"].duplicated().any():
        raise ValueError("A source prediction table contains duplicated Patient_ID values")
    if not frame["true_label"].isin([0, 1, 2]).all() or not frame["pred_label"].isin([0, 1, 2]).all():
        raise ValueError("Source labels must be encoded as 0, 1, and 2")
    probabilities = frame[["pred_prob_0", "pred_prob_1", "pred_prob_2"]].to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Source probabilities contain missing or infinite values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Source class probabilities do not sum to one")


def _anonymize(frame: pd.DataFrame, mapping: pd.Series, keep_index: bool = False) -> pd.DataFrame:
    output = frame.drop(columns=["Patient_ID"])
    output.insert(0, "sample_id", frame["Patient_ID"].map(mapping))
    if output["sample_id"].isna().any():
        raise ValueError("A source Patient_ID is absent from the anonymization mapping")
    if keep_index:
        output["_source_row"] = np.arange(len(output))
    return output


def _assert_consistent(base: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> None:
    merged = base[["Patient_ID", *columns]].merge(
        other[["Patient_ID", *columns]],
        on="Patient_ID",
        how="inner",
        validate="one_to_one",
        suffixes=("_base", "_other"),
    )
    if merged.empty:
        raise ValueError("Source tables do not share any Patient_ID values")
    for column in columns:
        if column == "Patient_ID":
            continue
        if isinstance(base[column].iloc[0], str):
            same = merged[f"{column}_base"].astype(str).eq(merged[f"{column}_other"].astype(str)).all()
        else:
            same = np.allclose(
                merged[f"{column}_base"].astype(float),
                merged[f"{column}_other"].astype(float),
            )
        if not same:
            raise ValueError(f"Source tables disagree on {column}")


def build_deidentified_data(source_root: Path, output_root: Path, seed: int = 42) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("The anonymized output directory must differ from the source directory")

    fusion_source = _read_source(source_root, "fusion_final/baseline_predictions.csv")
    radiomics_clinical_source = _read_source(source_root, "radiomics+clinical/baseline_predictions.csv")
    radiomics_source = _read_source(source_root, "radiomics/baseline_predictions.csv")
    habitat_source = _read_source(source_root, "habitat/baseline_predictions.csv")
    habitat_clinical_source = _read_source(source_root, "habitat+clinical/baseline_predictions.csv")
    ith_clinical_source = _read_source(source_root, "ITH+clinical/baseline_predictions.csv")

    source_ids = fusion_source["Patient_ID"].astype(str).tolist()
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("The fusion source contains duplicated Patient_ID values")
    # A deterministic permutation decorrelates sample order from raw identifiers without
    # retaining or publishing the linkage.
    permutation = np.random.default_rng(seed).permutation(len(source_ids))
    mapping = pd.Series(
        [f"S{i + 1:04d}" for i in permutation],
        index=pd.Index(source_ids, name="Patient_ID"),
    )

    fusion = _standard_prediction(fusion_source)
    radiomics_clinical = _standard_prediction(radiomics_clinical_source)
    habitat_clinical = _standard_prediction(habitat_clinical_source)
    for frame in (radiomics_clinical, habitat_clinical):
        _assert_consistent(fusion, frame, ["center", "true_label"])

    ith_probabilities = pd.DataFrame({
        "Patient_ID": ith_clinical_source["Patient_ID"].astype(str),
        "true_label": ith_clinical_source["True_Label"].astype(int),
        "pred_label": ith_clinical_source["Pred_Label"].astype(int),
        "pred_prob_0": ith_clinical_source["Pred_Prob_0"].astype(float),
        "pred_prob_1": ith_clinical_source["Pred_Prob_1"].astype(float),
        "pred_prob_2": ith_clinical_source["Pred_Prob_2"].astype(float),
    })
    _assert_valid_prediction(ith_probabilities)
    ith_clinical = ith_probabilities.merge(
        fusion[["Patient_ID", "center"]],
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )[["Patient_ID", "center", "true_label", "pred_label",
       "pred_prob_0", "pred_prob_1", "pred_prob_2"]]
    _assert_consistent(fusion, ith_clinical, ["center", "true_label"])

    for filename, _relative_path in PREDICTION_OUTPUTS.items():
        frame = {
            "fusion.csv": fusion,
            "radiomics_clinical.csv": radiomics_clinical,
            "habitat_clinical.csv": habitat_clinical,
            "ith_clinical.csv": ith_clinical,
        }[filename]
        destination = output_root / "predictions" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        _anonymize(frame, mapping).to_csv(destination, index=False)

    clinical = radiomics_clinical_source[["Patient_ID", "Tumor_Size", "Ulceration", "Invasion"]].copy()
    clinical = clinical.rename(columns={
        "Tumor_Size": "tumor_size",
        "Ulceration": "ulceration",
        "Invasion": "invasion",
    })
    clinical["Patient_ID"] = clinical["Patient_ID"].astype(str)
    radiomics_input = pd.DataFrame({
        "Patient_ID": radiomics_source["Patient_ID"].astype(str),
        "radiomics_label": radiomics_source["Pred_Label.1"].astype(int),
        "radiomics_prob_0": radiomics_source["Pred_0"].astype(float),
        "radiomics_prob_1": radiomics_source["Pred_1"].astype(float),
        "radiomics_prob_2": radiomics_source["Pred_2"].astype(float),
    })
    habitat_input = pd.DataFrame({
        "Patient_ID": habitat_source["Patient_ID"].astype(str),
        "habitat_label": habitat_source["Pred_Label.1"].astype(int),
        "habitat_prob_0": habitat_source["Pred_0"].astype(float),
        "habitat_prob_1": habitat_source["Pred_1"].astype(float),
        "habitat_prob_2": habitat_source["Pred_2"].astype(float),
    })
    frozen_fusion_input = pd.DataFrame({
        "Patient_ID": fusion_source["Patient_ID"].astype(str),
        "fusion_habitat_label": fusion_source["Hab_Pred_Label"].astype(int),
        "fusion_rad_prob_0": fusion_source["Rad_Pred_0"].astype(float),
        "fusion_rad_prob_1": fusion_source["Rad_Pred_1"].astype(float),
        "fusion_rad_prob_2": fusion_source["Rad_Pred_2"].astype(float),
        "ith_mean": fusion_source["ITH_Mean"].astype(float),
    })
    for prefix, frame in (("radiomics", radiomics_input), ("habitat", habitat_input)):
        probabilities = frame[[f"{prefix}_prob_{k}" for k in range(3)]].to_numpy(float)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{prefix} frozen branch probabilities contain invalid values")
    if not np.isfinite(frozen_fusion_input[["ith_mean"]].to_numpy(float)).all():
        raise ValueError("ITH_Mean contains missing or infinite values")

    features = fusion_source[["Patient_ID", "Center", "True_Label"]].copy()
    features.columns = ["Patient_ID", "center", "true_label"]
    features["Patient_ID"] = features["Patient_ID"].astype(str)
    features["true_label"] = features["true_label"].astype(int)
    for frame in (clinical, radiomics_input, habitat_input, frozen_fusion_input):
        features = features.merge(frame, on="Patient_ID", how="left", validate="one_to_one")
    if features.isna().any().any():
        raise ValueError("Model feature construction produced missing values")
    for prefix in ("radiomics", "habitat"):
        probabilities = features[[f"{prefix}_prob_{k}" for k in range(3)]].to_numpy(float)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Frozen {prefix} probabilities do not sum to one")
    features = features.drop(columns=["Patient_ID"])
    features.insert(0, "sample_id", fusion_source["Patient_ID"].astype(str).map(mapping))
    destination = output_root / "features" / "model_features.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)

    return {
        "patients": int(len(features)),
        "validation_patients": int(features["center"].isin([1, 4]).sum()),
        "prediction_tables": {name: int(len(frame)) for name, frame in {
            "fusion": fusion,
            "radiomics_clinical": radiomics_clinical,
            "habitat_clinical": habitat_clinical,
            "ith_clinical": ith_clinical,
        }.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Authorized results directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Repository data directory")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic ID-shuffle seed")
    args = parser.parse_args()
    summary = build_deidentified_data(args.source_root, args.output_root, args.seed)
    print(summary)


if __name__ == "__main__":
    main()
