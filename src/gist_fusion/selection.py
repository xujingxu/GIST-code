from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler


@dataclass
class MRMRFeatureSelector:
    """Training-only filtering and greedy F-relevance/Pearson-redundancy mRMR."""

    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    n_features: int = 22
    dropped_features_: list[str] = field(default_factory=list, init=False)
    selected_features_: list[str] = field(default_factory=list, init=False)
    scaler_: StandardScaler = field(default_factory=StandardScaler, init=False)

    def _filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        diagnostics = [c for c in frame if "diagnostics" in c.lower()]
        variances = frame.var(numeric_only=True)
        low_variance = variances[variances < self.variance_threshold].index.tolist()
        filtered = frame.drop(columns=list(set(diagnostics + low_variance)))
        if filtered.shape[1] < 2:
            raise ValueError("Fewer than two features remain after variance filtering")

        correlation = filtered.corr().abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        to_drop: set[str] = set()
        for column in upper.columns:
            for feature in upper.index[upper[column] > self.correlation_threshold]:
                if filtered[column].var() >= filtered[feature].var():
                    to_drop.add(feature)
                else:
                    to_drop.add(column)
        self.dropped_features_ = sorted(set(diagnostics + low_variance) | to_drop)
        return filtered.drop(columns=sorted(to_drop), errors="ignore")

    def fit_select(self, features: pd.DataFrame, labels: np.ndarray) -> list[str]:
        """Fit exclusively on training data and return the ordered selected features."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame with named columns")
        numeric = features.select_dtypes(include=[np.number]).dropna(axis=1, how="any")
        if len(numeric) != len(labels):
            raise ValueError("features and labels must have the same number of rows")
        filtered = self._filter(numeric)
        k = min(self.n_features, filtered.shape[1])
        scaled = self.scaler_.fit_transform(filtered.to_numpy(float))
        f_values, _ = f_classif(scaled, labels)
        maximum = np.nanmax(f_values)
        relevance = f_values / maximum if maximum > 0 else np.zeros_like(f_values)

        selected: list[int] = [int(np.argmax(relevance))]
        candidates = set(range(filtered.shape[1])) - set(selected)
        while len(selected) < k and candidates:
            best_index, best_score = None, -np.inf
            for index in sorted(candidates):
                redundancy = float(np.mean([
                    abs(np.corrcoef(scaled[:, index], scaled[:, chosen])[0, 1])
                    for chosen in selected
                ]))
                if np.isnan(redundancy):
                    redundancy = 0.0
                score = relevance[index] - redundancy
                if score > best_score:
                    best_index, best_score = index, score
            selected.append(best_index)
            candidates.remove(best_index)
        self.selected_features_ = filtered.columns[selected].tolist()
        self._filtered_names = filtered.columns.tolist()
        return self.selected_features_

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if not self.selected_features_:
            raise RuntimeError("Call fit_select before transform")
        return self.scaler_.transform(features[self._filtered_names].to_numpy(float))
