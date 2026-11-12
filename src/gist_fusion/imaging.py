from __future__ import annotations

from pathlib import Path

import numpy as np


def _simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "Imaging functions require the optional dependencies: "
            "pip install -r requirements-imaging.txt"
        ) from error
    return sitk


def zscore_roi(image: np.ndarray, mask: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize the entire image using statistics from mask > 0."""
    values = image[mask > 0]
    if values.size == 0:
        raise ValueError("Cannot normalize an empty ROI")
    return (image - values.mean()) / (values.std() + epsilon)


def resample_to_isotropic(image, mask=None, spacing_mm: float = 1.0):
    """B-spline image resampling and nearest-neighbor mask resampling."""
    sitk = _simpleitk()
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    target_spacing = (spacing_mm,) * 3
    target_size = [
        int(round(original_size[i] * original_spacing[i] / spacing_mm))
        for i in range(3)
    ]

    def resample(source, interpolation):
        result = sitk.Resample(
            source,
            target_size,
            sitk.Transform(),
            interpolation,
            source.GetOrigin(),
            target_spacing,
            source.GetDirection(),
            0,
        )
        return result

    resampled_image = resample(image, sitk.sitkBSpline)
    resampled_mask = resample(mask, sitk.sitkNearestNeighbor) if mask is not None else None
    return resampled_image, resampled_mask


def extract_whole_tumor_features(image_path: str | Path, mask_path: str | Path) -> dict:
    """Extract all enabled PyRadiomics features after 1-mm resampling and ROI z-scoring."""
    sitk = _simpleitk()
    try:
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError as error:
        raise ImportError("Install pyradiomics==3.0.1 to extract radiomic features") from error
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    image, mask = resample_to_isotropic(image, mask, spacing_mm=1.0)
    normalized = zscore_roi(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(mask))
    normalized_image = sitk.GetImageFromArray(normalized.astype(np.float32))
    normalized_image.CopyInformation(image)
    extractor = RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    return dict(extractor.execute(normalized_image, mask))


def cluster_probability_map(probability: np.ndarray, mask: np.ndarray, seed: int = 42):
    """Per-patient two-habitat K-means, ordered from low to high probability."""
    from sklearn.cluster import KMeans

    roi = mask > 0
    if not roi.any():
        raise ValueError("The tumor mask is empty")
    values = probability[roi]
    scale = values.max() - values.min()
    if scale == 0:
        raise ValueError("Cannot form two habitats from a constant probability map")
    normalized = ((values - values.min()) / scale).reshape(-1, 1)
    model = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(normalized)
    order = np.argsort(model.cluster_centers_.ravel())
    remap = np.zeros(2, dtype=np.uint8)
    remap[order] = np.arange(2, dtype=np.uint8)
    labels = np.zeros(probability.shape, dtype=np.uint8)
    labels[roi] = remap[model.labels_] + 1
    return labels
