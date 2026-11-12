# Data Dictionary

## Cohort and labels

- Total patients: 701
- Training centers: 0, 2, 3, 5; patients: 540
- Validation centers: 1, 4; patients: 161
- Validation class counts: low = 103, intermediate = 41, high = 17
- Risk labels: `0 = low`, `1 = intermediate`, `2 = high`

Center numbers are coded site identifiers. They are not hospital names and are not patient identifiers.

## `data/features/model_features.csv`

One row per patient. All 701 rows include the frozen model inputs needed to retrain the four final classifiers.

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | Random deidentified study code, `S0001`–`S0701`; unique |
| `center` | integer | Coded enrollment center; `{0,2,3,5}` train, `{1,4}` validation |
| `true_label` | integer | Pathological risk label: 0 low, 1 intermediate, 2 high |
| `radiomics_label` | integer | Frozen predicted label from the independent Radiomics branch |
| `radiomics_prob_0` | float | Frozen Radiomics low-risk probability |
| `radiomics_prob_1` | float | Frozen Radiomics intermediate-risk probability |
| `radiomics_prob_2` | float | Frozen Radiomics high-risk probability |
| `habitat_label` | integer | Frozen predicted label from the independent Habitat branch |
| `habitat_prob_0` | float | Frozen Habitat low-risk probability |
| `habitat_prob_1` | float | Frozen Habitat intermediate-risk probability |
| `habitat_prob_2` | float | Frozen Habitat high-risk probability |
| `ith_mean` | float | Repeated-measure mean ITH value used by ITH-Clinical and Fusion |
| `tumor_size` | numeric | Maximum tumor diameter/size from the clinical feature table |
| `ulceration` | integer/binary | Surface-ulceration clinical variable |
| `invasion` | integer/binary | Invasive-margin clinical variable |
| `fusion_habitat_label` | integer | Habitat label frozen with the final Fusion run |
| `fusion_rad_prob_0` | float | Radiomics probability frozen with the final Fusion run |
| `fusion_rad_prob_1` | float | Radiomics probability frozen with the final Fusion run |
| `fusion_rad_prob_2` | float | Radiomics probability frozen with the final Fusion run |

### Important model-input provenance

The Fusion model must be retrained from `fusion_habitat_label`, `fusion_rad_prob_*`, and `ith_mean`. Do not substitute `habitat_label` or `radiomics_prob_*` for these columns: the preserved Fusion input reflects the exact historical late-fusion run, while the independently exported branch tables are not necessarily the same algorithm-version snapshot.

## `data/predictions/*.csv`

All prediction tables use this schema:

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | Deidentified study code; unique within each table |
| `center` | integer | Coded enrollment center |
| `true_label` | integer | Pathological risk label |
| `pred_label` | integer | Model-predicted risk label |
| `pred_prob_0` | float | Predicted low-risk probability |
| `pred_prob_1` | float | Predicted intermediate-risk probability |
| `pred_prob_2` | float | Predicted high-risk probability |

Available files:

| File | Rows | Scope |
|---|---:|---|
| `fusion.csv` | 701 | Final Fusion predictions for the full cohort |
| `radiomics_clinical.csv` | 701 | Final Radiomics-Clinical predictions for the full cohort |
| `habitat_clinical.csv` | 161 | Final Habitat-Clinical predictions retained for the independent validation cohort |
| `ith_clinical.csv` | 701 | Final ITH-Clinical predictions for the full cohort |

`habitat_clinical.csv` is intentionally validation-only. No training predictions were fabricated to make the table appear complete.

## Data derivation

The author-side helper `scripts/prepare_deidentified_data.py` documents how the released CSV files are built from the authorized source result tables. Public users do not need to run it. The helper:

1. validates labels, centers, uniqueness, and probability normalization;
2. checks cross-table agreement of shared patient records and true labels;
3. applies a deterministic random permutation from original patient IDs to `S####` sample IDs; and
4. writes no original patient IDs or identifier mapping file.

The current release retains source probabilities at their original floating-point precision.

## Privacy statement

The repository contains no raw images, segmentation masks, original patient identifiers, acquisition dates, hospital names, or identifier map. The study-code mapping is not published. Center labels and clinical values remain potentially sensitive derived data; release must be approved through the applicable ethics, institutional, and journal data-sharing procedures.
