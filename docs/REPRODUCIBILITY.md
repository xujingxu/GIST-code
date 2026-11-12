# Reproducibility Notes

## What this repository reproduces

### Route A — exact reported statistics

The `reproduce` command evaluates the released frozen validation predictions and independently recomputes class-wise, macro, weighted, and micro AUCs, bootstrap confidence intervals, confusion matrices, and Fusion-versus-Radiomics-Clinical continuous NRI/IDI.

```bash
gist-fusion --config configs/default.yaml reproduce \
  --output runs/paper_reproduction \
  --bootstrap 2000
```

Expected point estimates:

| Metric | Expected value |
|---|---:|
| Fusion low-risk AUC | 0.887764 |
| Fusion intermediate-risk AUC | 0.779878 |
| Fusion high-risk AUC | 0.903391 |
| Fusion macro AUC | 0.857011 |
| Fusion weighted AUC | 0.861940 |
| Fusion micro AUC | 0.905289 |
| Macro continuous NRI, Fusion vs Radiomics-Clinical | 0.943307 |
| Macro IDI, Fusion vs Radiomics-Clinical | 0.120380 |

### Route B — final-model retraining

The `train` command standardizes only the training cohort, fits each final model with the paper’s final hyperparameters, and evaluates the untouched center-level validation cohort.

```bash
gist-fusion --config configs/default.yaml train \
  --output runs/retrained_models \
  --bootstrap 2000
```

Expected behavior in the locked environment:

| Final model | Frozen weighted AUC | Retrained weighted AUC | Absolute difference |
|---|---:|---:|---:|
| Radiomics-Clinical | 0.848471 | 0.849878 | 0.001407 |
| Habitat-Clinical | 0.838826 | 0.836802 | 0.002024 |
| ITH-Clinical | 0.812117 | 0.812172 | 0.000055 |
| Fusion | 0.861940 | 0.861940 | 0.000000 |

The Fusion stage is exactly reproducible from its preserved frozen inputs and final hyperparameters. Small differences in the other three stages reflect historical branch-pipeline implementation differences, not validation-patient leakage. Claims should therefore distinguish “paper statistics reproduction” from “final-model retraining.”

## Locked validation environment

The repository was exercised with:

- Python 3.12.15
- NumPy 2.2.6
- pandas 2.3.3
- SciPy 1.15.2
- scikit-learn 1.7.2
- XGBoost 3.0.5
- joblib 1.5.2
- PyYAML 6.0.3

`requirements.txt` pins the same numerical-library versions. `pyproject.toml` permits compatible maintenance ranges, while `requirements.txt` is the recommended environment for audit and review.

## Determinism and validation isolation

- All cohort assignment is by coded center, never by random patient split.
- The validation cohort is not used for model fitting, scaling, feature selection, or algorithm selection.
- Model seeds come from `experiment.seed` in `configs/default.yaml`.
- AUC bootstrap replicates use seed 42.
- Paired NRI/IDI bootstrap replicates use seed 43 so the paired procedure does not reuse the AUC random stream.
- `MRMRFeatureSelector` fits variance filtering, correlation filtering, scaling, relevance, redundancy, and selected features exclusively on the training cohort.
- Habitat-Clinical prediction data are validation-only; no missing training rows are synthesized.

## Tests

```bash
python -m pytest
python -m ruff check src scripts tests
```

The test suite verifies representative formulas with hand-calculated values and checks that validation patients cannot appear in training output.

## Known limitations

1. **No end-to-end image-level reproduction.** The repository does not include CT images or masks. It also cannot independently rebuild every intermediate radiomics or habitat branch output because branch-model training implementations and intermediate snapshots are not all part of this release.
2. **Optional imaging helpers are partial utilities.** They document resampling, normalization, PyRadiomics extraction, habitat clustering, and the ITH connected-component score, but they are not represented as the complete validated source pipeline.
3. **Binary clinical encoding.** `ulceration` and `invasion` are released only as model-ready encoded variables. Their original categorical source semantics should be described in the manuscript and supplement.
4. **Legal and ethical approval.** Deidentified derived data are still data derived from human participants. Public release requires author confirmation and compliance with the approving ethics committee, institutions, journal policy, and any data-sharing agreements.

## Author-side data preparation

Authorized study personnel can regenerate the CSV files with:

```bash
python scripts/prepare_deidentified_data.py \
  --source-root /path/to/authorized/results \
  --output-root data \
  --seed 42
```

This command requires the original authorized result tables, never writes an identifier map, and will fail on duplicate IDs, inconsistent shared labels, invalid probabilities, or missing values.
