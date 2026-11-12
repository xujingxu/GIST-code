# GIST-Multiscale-Reproducibility

Repository: https://github.com/xujingxu/GIST-code

Reproducible code and deidentified model-level data for **“Multiscale CT Fusion for Preoperative Three-Class Risk Stratification of Gastric Gastrointestinal Stromal Tumors: A Multicenter Study.”**

This repository provides two explicit reproduction routes:

1. **Exact paper-statistics reproduction** from frozen, deidentified validation predictions.
2. **Final-model retraining** from frozen, deidentified model inputs with the paper’s final hyperparameters.

Raw CT images, segmentation masks, raw clinical identifiers, and radiology reports are **not** distributed. The public imaging helpers are optional and are not a complete end-to-end replacement for the original imaging pipeline.

## Scope

The study enrolled 701 patients from six centers. Center-level partitioning prevents patient leakage:

| Cohort | Centers | Patients |
|---|---:|---:|
| Training | 0, 2, 3, 5 | 540 |
| Independent validation | 1, 4 | 161 |

Risk labels are encoded as `0 = low`, `1 = intermediate`, and `2 = high`.

The supplied feature tables are **frozen branch-model outputs** rather than raw radiomic feature matrices. This is intentional: it makes the final late-fusion stage and all reported validation metrics reproducible without exposing images or identifiable source data. It also preserves the exact historical inputs used by the Fusion model. See [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for the model-specific input provenance.

## Installation

Use Python 3.10, 3.11, or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a dependency-only environment, install `requirements-dev.txt` instead:

```bash
python -m pip install -r requirements-dev.txt
```

## Reproduce the reported validation statistics

```bash
gist-fusion --config configs/default.yaml reproduce \
  --output runs/paper_reproduction \
  --bootstrap 2000
```

Key independently validated outputs are:

- Fusion weighted-average validation AUC: **0.8619397**
- Fusion macro continuous NRI versus Radiomics-Clinical: **0.943307** (95% CI 0.702030–1.201331)
- Fusion macro IDI versus Radiomics-Clinical: **0.120380** (95% CI 0.085065–0.156638)

Generated files include:

- `validation_auc.csv`
- `validation_confusion_matrices.csv`
- `fusion_vs_radiomics_nri_idi.csv`
- `summary.json`

## Retrain the four final classifiers

```bash
gist-fusion --config configs/default.yaml train \
  --output runs/retrained_models \
  --bootstrap 2000
```

This command trains Radiomics-Clinical, Habitat-Clinical, ITH-Clinical, and Fusion from the frozen inputs listed in `configs/default.yaml`. Each model directory contains the fitted model, scaler, feature manifest, training predictions, validation predictions, and summary metrics.

In the locked test environment described below, Fusion retraining exactly matched the frozen Fusion weighted-average validation AUC. The other three retrained paths differed by no more than approximately 0.002 AUC from their frozen model outputs because the released inputs preserve final model inputs but not every historical preprocessing implementation detail of the branch pipelines.

## Run tests and linting

```bash
python -m pytest
python -m ruff check src scripts tests
```

The test suite covers center isolation, prediction-table validation, AUC and bootstrap determinism, hand-calculated continuous NRI/IDI, training-only mRMR behavior, the connected-component ITH formula, and both supported classifier families.

## Repository layout

```text
configs/                 Experiment partition, data paths, features, and final hyperparameters
data/features/           Deidentified frozen model inputs for final-model training
data/predictions/        Deidentified frozen model predictions
docs/                    Data provenance and reproducibility notes
scripts/                 Author-side data preparation and command wrappers
src/gist_fusion/         Installable analysis package
tests/                   Automated tests
```

## Optional imaging utilities

The package includes optional helpers for lesion normalization, isotropic resampling, whole-tumor PyRadiomics extraction, two-habitat clustering, and ITH-score calculation. Install them with:

```bash
python -m pip install -e ".[imaging]"
```

These helpers do **not** constitute a complete, independently validated image-to-final-prediction pipeline. The original repository cannot regenerate the radiomics and habitat branch outputs end-to-end without the withheld images, masks, branch-specific intermediate data, and branch-model training implementations.

## Data and privacy notice

- `sample_id` values (`S0001`–`S0701`) are random study codes.
- No original `Patient_ID`, name, date, image, mask, or identifier map is included.
- The identifier map used to create these files is deliberately not retained in the repository.
- Even deidentified feature and prediction tables should be released only after the authors confirm that doing so complies with the approving IRB/ethics committee, institutional policy, journal requirements, and any applicable data-sharing agreements.

## Citation

A `CITATION.cff` file is included. Before creating an archival release, add the repository URL and, once available, the manuscript DOI or Zenodo DOI.

## License

MIT License. The license choice should be confirmed by all manuscript authors before public release.
