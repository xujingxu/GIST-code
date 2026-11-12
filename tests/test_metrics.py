import numpy as np
import pandas as pd
import pytest

from gist_fusion.metrics import (
    auc_values,
    bootstrap_estimates,
    confusion_rows,
    continuous_nri_idi,
    paired_by_sample,
)


def test_perfect_three_class_probabilities_have_unit_auc() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.eye(3)[labels]
    values = auc_values(labels, probabilities)
    assert values == pytest.approx({
        "class_0_auc": 1.0,
        "class_1_auc": 1.0,
        "class_2_auc": 1.0,
        "macro_auc": 1.0,
        "weighted_auc": 1.0,
        "micro_auc": 1.0,
    })


def test_auc_requires_all_three_classes() -> None:
    assert auc_values(np.array([0, 0, 1, 1]), np.ones((4, 3)) / 3) is None


def test_bootstrap_intervals_are_deterministic() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    first_point, first_intervals, first_n = bootstrap_estimates(
        labels, probabilities, auc_values, replicates=25, seed=7
    )
    second_point, second_intervals, second_n = bootstrap_estimates(
        labels, probabilities, auc_values, replicates=25, seed=7
    )
    assert first_point == second_point
    assert first_intervals == second_intervals
    assert first_n == second_n
    assert 0 < first_n <= 25


def test_bootstrap_requires_positive_replicates() -> None:
    with pytest.raises(ValueError, match="positive"):
        bootstrap_estimates(
            np.array([0, 1, 2]),
            np.eye(3),
            auc_values,
            replicates=0,
        )


def test_continuous_nri_and_idi_match_hand_calculation() -> None:
    labels = np.array([0, 1, 2])
    old = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2],
        [0.2, 0.1, 0.7],
    ])
    new = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
    ])
    values = continuous_nri_idi(labels, old, new)
    assert values["class_0_nri"] == pytest.approx(2.0)
    assert values["class_1_nri"] == pytest.approx(1.0)
    assert values["class_2_nri"] == pytest.approx(0.0)
    assert values["class_0_idi"] == pytest.approx(0.2)
    assert values["class_1_idi"] == pytest.approx(0.1)
    assert values["class_2_idi"] == pytest.approx(0.0)
    assert values["macro_nri"] == pytest.approx(1.0)
    assert values["macro_idi"] == pytest.approx(0.3 / 3.0)


def test_continuous_nri_returns_a_complete_nan_key_set_for_a_missing_class() -> None:
    values = continuous_nri_idi(
        np.array([0, 0]),
        np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]),
        np.array([[0.7, 0.2, 0.1], [0.8, 0.1, 0.1]]),
    )
    expected_keys = {
        *(f"class_{index}_{measure}" for index in range(3) for measure in ("nri", "idi")),
        "macro_nri",
        "macro_idi",
    }
    assert set(values) == expected_keys
    assert all(np.isnan(value) for value in values.values())


def test_confusion_rows_use_paper_class_order() -> None:
    rows = confusion_rows("Fusion", np.array([0, 1, 1, 2]), np.array([0, 2, 1, 0]))
    assert rows == [
        {
            "model": "Fusion",
            "true_class": "Low risk",
            "predicted_low": 1,
            "predicted_intermediate": 0,
            "predicted_high": 0,
        },
        {
            "model": "Fusion",
            "true_class": "Intermediate risk",
            "predicted_low": 0,
            "predicted_intermediate": 1,
            "predicted_high": 1,
        },
        {
            "model": "Fusion",
            "true_class": "High risk",
            "predicted_low": 1,
            "predicted_intermediate": 0,
            "predicted_high": 0,
        },
    ]


def test_pairing_requires_identical_true_labels() -> None:
    first = pd.DataFrame({
        "sample_id": ["S0001", "S0002"],
        "true_label": [0, 1],
        "pred_label": [0, 1],
        "pred_prob_0": [0.8, 0.1],
        "pred_prob_1": [0.1, 0.8],
        "pred_prob_2": [0.1, 0.1],
    })
    second = first.copy()
    second.loc[0, "true_label"] = 1
    with pytest.raises(ValueError, match="True labels differ"):
        paired_by_sample(first, second)
