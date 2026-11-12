import joblib
import numpy as np
import pandas as pd

from gist_fusion.models import fit_final_model, save_artifacts


def cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    train = pd.DataFrame({
        "sample_id": [f"S{index + 1:04d}" for index in range(24)],
        "center": np.repeat([0, 2], 12),
        "true_label": np.tile([0, 1, 2], 8),
        "branch_label": np.tile([0, 1, 2], 8),
        "tumor_size": rng.uniform(2, 12, 24),
        "ulceration": rng.integers(0, 2, 24),
        "invasion": rng.integers(0, 2, 24),
        "ith_mean": rng.uniform(0.2, 0.9, 24),
    })
    validation = pd.DataFrame({
        "sample_id": [f"V{index + 1:04d}" for index in range(9)],
        "center": np.repeat([1, 4], [5, 4]),
        "true_label": np.tile([0, 1, 2], 3),
        "branch_label": np.tile([0, 1, 2], 3),
        "tumor_size": rng.uniform(2, 12, 9),
        "ulceration": rng.integers(0, 2, 9),
        "invasion": rng.integers(0, 2, 9),
        "ith_mean": rng.uniform(0.2, 0.9, 9),
    })
    return train, validation


def test_fit_final_model_supports_xgboost_and_svm() -> None:
    train, validation = cohort()
    features = ["branch_label", "tumor_size", "ulceration", "invasion", "ith_mean"]
    specifications = {
        "xgboost": ("xgboost", {
            "n_estimators": 8,
            "max_depth": 2,
            "learning_rate": 0.1,
            "eval_metric": "mlogloss",
            "n_jobs": 1,
        }),
        "svm": ("svm", {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
        }),
    }
    for _model_type, (kind, params) in specifications.items():
        model, scaler, train_predictions, validation_predictions = fit_final_model(
            train, validation, features, kind, params, seed=42
        )
        assert len(model.classes_) == 3
        assert scaler.n_features_in_ == len(features)
        assert len(train_predictions) == len(train)
        assert len(validation_predictions) == len(validation)
        for predictions in (train_predictions, validation_predictions):
            probabilities = predictions[["pred_prob_0", "pred_prob_1", "pred_prob_2"]].to_numpy(float)
            assert np.allclose(probabilities.sum(axis=1), 1.0)
            assert predictions["pred_label"].isin([0, 1, 2]).all()


def test_save_artifacts_writes_model_scaler_and_feature_manifest(tmp_path) -> None:
    train, validation = cohort()
    features = ["branch_label", "tumor_size", "ulceration", "invasion", "ith_mean"]
    model, scaler, _, _ = fit_final_model(
        train,
        validation,
        features,
        "xgboost",
        {
            "n_estimators": 6,
            "max_depth": 2,
            "learning_rate": 0.1,
            "eval_metric": "mlogloss",
            "n_jobs": 1,
        },
        seed=42,
    )
    save_artifacts(tmp_path, "Radiomics-Clinical", model, scaler, features)
    assert joblib.load(tmp_path / "radiomics_clinical_model.joblib") is not None
    assert joblib.load(tmp_path / "radiomics_clinical_scaler.joblib") is not None
    assert (tmp_path / "radiomics_clinical_features.txt").read_text(encoding="utf-8").splitlines() == features
