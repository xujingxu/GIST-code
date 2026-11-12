# Contributing

Contributions should preserve reproducibility, patient privacy, and the separation between the frozen paper evaluation and optional imaging utilities.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src scripts tests
```

## Change guidelines

- Keep all cohort assignment center-based; do not introduce random patient-level splits.
- Fit scalers, feature selectors, and models only on the training cohort.
- Validate all supplied tables for unique sample IDs, valid labels, finite values, and normalized probabilities.
- Add or update tests for numerical formulas and data-validation behavior.
- Do not commit raw images, masks, source clinical tables, patient identifiers, identifier maps, fitted model binaries, or local run directories.
- Document any change that alters expected metrics, seeds, data schemas, or the scope of image-level reproducibility.

## Reporting issues

Include the command, configuration, Python version, operating system, and package versions. Do not attach identifiable patient data or unauthorized source data.
