import numpy as np
import pandas as pd
import pytest

from gist_fusion.selection import MRMRFeatureSelector


def training_frame() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    informative = rng.normal(size=(30, 3))
    copied = informative[:, [0, 1]] + rng.normal(scale=0.001, size=(30, 2))
    low_variance = np.full((30, 1), 3.14159)
    frame = pd.DataFrame(
        np.column_stack([informative, copied, low_variance]),
        columns=[
            "signal_0",
            "signal_1",
            "noise",
            "signal_0_copy",
            "signal_1_copy",
            "constant_feature",
        ],
    )
    labels = (informative[:, 0] + 0.5 * informative[:, 1] > 0).astype(int)
    return frame, labels


def test_mrmr_fits_all_statistics_on_training_data_only() -> None:
    train, labels = training_frame()
    selector = MRMRFeatureSelector(variance_threshold=0.01, correlation_threshold=0.95, n_features=3)
    selected = selector.fit_select(train, labels)
    assert len(selected) == 3
    assert "constant_feature" not in selected
    assert len(set(selected)) == len(selected)
    transformed = selector.transform(train)
    assert transformed.shape == (len(train), len(selected))


def test_mrmr_requires_matching_row_counts() -> None:
    features, labels = training_frame()
    with pytest.raises(ValueError, match="same number of rows"):
        MRMRFeatureSelector().fit_select(features.iloc[:-1], labels)


def test_mrmr_transform_requires_fit() -> None:
    features, _ = training_frame()
    with pytest.raises(RuntimeError, match="fit_select"):
        MRMRFeatureSelector().transform(features)
