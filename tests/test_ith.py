import numpy as np
import pytest

from gist_fusion.ith import ith_score, largest_tumor_slice, repeated_ith


def test_ith_score_matches_connected_component_formula() -> None:
    # With the default 8-neighbour connectivity, each cluster has two components of
    # area two; total tumor area is eight pixels, so ITH = 1 - (2/2 + 2/2) / 8 = 0.75.
    label_map = np.array([
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 2, 2],
    ])
    assert ith_score(label_map) == pytest.approx(0.75)


def test_ith_connectivity_two_uses_eight_neighbour_components() -> None:
    label_map = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
    ])
    assert ith_score(label_map, connectivity=1, min_area=0) == pytest.approx(8.0 / 9.0)
    assert ith_score(label_map, connectivity=2, min_area=0) == pytest.approx(0.0)


def test_small_tumor_components_follow_reference_area_threshold() -> None:
    label_map = np.array([[1, 0, 2]])
    assert ith_score(label_map, min_area=200, small_region_threshold=1) == pytest.approx(1.0)


def test_ith_rejects_empty_or_wrong_dimensional_maps() -> None:
    with pytest.raises(ValueError, match="2-D"):
        ith_score(np.zeros((2, 2, 2), dtype=int))
    with pytest.raises(ValueError, match="no tumor"):
        ith_score(np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match="connectivity"):
        ith_score(np.ones((2, 2), dtype=int), connectivity=3)


def test_largest_tumor_slice_uses_maximum_in_slice_area() -> None:
    mask = np.zeros((3, 4, 4), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[2, :2, :2] = 1
    assert largest_tumor_slice(mask) == 2


def test_repeated_ith_uses_fixed_seeds_and_returns_replicates() -> None:
    rng = np.random.default_rng(4)
    features = np.column_stack([
        np.repeat([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 4),
        rng.normal(size=24),
    ])
    mean, scores = repeated_ith(features, (4, 6), n_clusters=6, repetitions=3, seed=42)
    mean_again, scores_again = repeated_ith(features, (4, 6), n_clusters=6, repetitions=3, seed=42)
    assert len(scores) == 3
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert mean == pytest.approx(np.mean(scores))
    assert scores == scores_again
    assert mean == pytest.approx(mean_again)
