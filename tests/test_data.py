import pandas as pd
import pytest

from gist_fusion.data import (
    load_predictions,
    select_validation,
    split_by_center,
    validate_prediction_frame,
)


def prediction_frame(centers=(0, 1, 2, 3)) -> pd.DataFrame:
    rows = []
    for index, center in enumerate(centers):
        true_label = index % 3
        probability = [0.15, 0.15, 0.15]
        probability[true_label] = 0.7
        rows.append({
            "sample_id": f"S{index + 1:04d}",
            "center": center,
            "true_label": true_label,
            "pred_label": true_label,
            "pred_prob_0": probability[0],
            "pred_prob_1": probability[1],
            "pred_prob_2": probability[2],
        })
    return pd.DataFrame(rows)


def test_center_split_has_no_patient_leakage() -> None:
    train, validation = split_by_center(prediction_frame(), [0, 2], [1, 3])
    assert train["sample_id"].tolist() == ["S0001", "S0003"]
    assert validation["sample_id"].tolist() == ["S0002", "S0004"]
    assert set(train["sample_id"]).isdisjoint(validation["sample_id"])


def test_center_split_rejects_unassigned_centers() -> None:
    with pytest.raises(ValueError, match="unassigned"):
        split_by_center(prediction_frame(), [0], [1])


def test_select_validation_accepts_full_or_validation_only_tables() -> None:
    full = prediction_frame()
    validation_only = full[full["center"].isin([1, 3])].copy()
    for frame in (full, validation_only):
        selected = select_validation(frame, [1, 3], allowed_centers=[0, 1, 2, 3])
        assert selected["sample_id"].tolist() == ["S0002", "S0004"]


def test_prediction_validator_rejects_invalid_probability_rows() -> None:
    frame = prediction_frame((0,))
    frame.loc[0, "pred_prob_2"] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_prediction_frame(frame)


def test_prediction_validator_rejects_probability_rows_not_summing_to_one() -> None:
    frame = prediction_frame((0,))
    frame.loc[0, ["pred_prob_0", "pred_prob_1", "pred_prob_2"]] = [0.8, 0.2, 0.2]
    with pytest.raises(ValueError, match="sum to 1"):
        validate_prediction_frame(frame)


def test_load_predictions_reads_valid_csv(tmp_path) -> None:
    path = tmp_path / "predictions.csv"
    prediction_frame().to_csv(path, index=False)
    loaded = load_predictions(path)
    assert len(loaded) == 4
    assert loaded["sample_id"].is_unique
