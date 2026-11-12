from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans


def largest_tumor_slice(mask: np.ndarray) -> int:
    if mask.ndim != 3:
        raise ValueError("Expected a 3D mask")
    areas = np.sum(mask > 0, axis=(1, 2))
    if areas.sum() == 0:
        raise ValueError("The tumor mask is empty")
    return int(np.argmax(areas))


def cluster_pixels(
    local_features: np.ndarray,
    n_clusters: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Cluster pixels within one lesion; labels are lesion-specific and unordered."""
    features = np.asarray(local_features, dtype=float)
    if features.ndim != 2:
        raise ValueError("local_features must have shape (n_pixels, n_features)")
    if len(features) < n_clusters:
        raise ValueError("The lesion has fewer pixels than requested clusters")
    if not np.isfinite(features).all():
        raise ValueError("Local features contain missing or infinite values")
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return model.fit_predict(features)


def ith_score(
    label_map: np.ndarray,
    background_label: int = 0,
    connectivity: int = 2,
    min_area: int = 200,
    small_region_threshold: int = 1,
) -> float:
    """ITH = 1 - Σ_i S_i,max / (n_i S_total), following the ITHscore definition.

    Tumor pixels are every label different from ``background_label``; cluster IDs need not
    be contiguous. Connectivity 2 is the reference package's 8-neighbour 2-D definition.
    """
    labels = np.asarray(label_map)
    if labels.ndim != 2:
        raise ValueError("Expected a 2-D lesion label map")
    tumor = labels != background_label
    total = int(tumor.sum())
    if total == 0:
        raise ValueError("The label map contains no tumor pixels")
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 (4-neighbour) or 2 (8-neighbour)")
    structure = ndimage.generate_binary_structure(2, connectivity)
    penalty = 0.0
    for cluster in np.unique(labels[tumor]):
        components, count = ndimage.label(labels == cluster, structure=structure)
        max_area = 0
        for component in range(1, count + 1):
            component_area = int((components == component).sum())
            if total <= min_area and component_area <= small_region_threshold:
                count -= 1
                if count == 0:
                    break
            elif component_area > max_area:
                max_area = component_area
        if count > 0:
            penalty += max_area / count
    return float(1.0 - penalty / total)


def repeated_ith(
    local_features: np.ndarray,
    label_map_shape: tuple[int, int],
    n_clusters: int = 6,
    repetitions: int = 10,
    seed: int = 42,
    connectivity: int = 2,
    min_area: int = 200,
    small_region_threshold: int = 1,
) -> tuple[float, list[float]]:
    """Repeat lesion clustering and return the mean ITH score and its replicates."""
    scores = []
    for repeat in range(repetitions):
        clustered = cluster_pixels(local_features, n_clusters=n_clusters, seed=seed + repeat)
        scores.append(ith_score(
            clustered.reshape(label_map_shape),
            background_label=0,
            connectivity=connectivity,
            min_area=min_area,
            small_region_threshold=small_region_threshold,
        ))
    return float(np.mean(scores)), scores
