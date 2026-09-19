# Bridging Gaps in Paleoclimatological Records: Insights from AI on Speleothem Data

Code, processed data, and results accompanying the manuscript submitted to *Ecological Informatics*.

This repository contains the pipeline used to evaluate machine learning and deep learning approaches to imputing missing geochemical proxy measurements in speleothem records, using the SISALv3 database enriched with external climate covariates.

---

## Overview

Speleothem records frequently contain gaps, arising either from genuine growth interruptions or from analytical and logistical limitations in sampling. This work evaluates whether machine learning models can usefully reconstruct those missing measurements, and establishes the conditions under which they can and cannot.

Five model architectures are benchmarked across four target proxies under two complementary leakage-free evaluation protocols, with explicit comparison against standard interpolation baselines.

**Target proxies:** δ¹⁸O, δ¹³C, Mg/Ca, Sr/Ca

**Models:** Random Forest, XGBoost, CatBoost, Bidirectional LSTM, Transformer

**Evaluation protocols:**

| Protocol | Question addressed | Held out |
|---|---|---|
| **LOCO** (Leave-One-Cave-Out) | How well does a model generalize to a cave with no measurements of its own? | All samples from one cave site, in turn |
| **WCBG** (Within-Cave Block-Gap) | How well can a gap be filled inside a record that already has surrounding measurements? | A contiguous block of samples within one entity |

The two protocols bound the practical use case at either end. Under LOCO, standard interpolation cannot be applied at all, as there are no anchor points in an unmeasured record; this is where the models provide their advantage. Under WCBG, where a gap is bounded by measurements on both sides, linear interpolation is competitive and the models do not significantly outperform it. The difference between the two regimes is itself a result about how far each proxy is controlled locally rather than regionally.

---

## Repository contents

```
.
├── notebooks/
│   └── 001_pipeline_train_evaluate.ipynb
│         Data assembly, enrichment, feature engineering, model training,
│         LOCO and WCBG evaluation, baseline comparisons, significance tests
│
├── data/
│   ├── master_table_enriched.parquet     Assembled cohort, 18,004 × 49
│   ├── cave_cohort_41sites.csv           Cave sites used in training
│   └── pipeline_metadata.json            Run configuration and feature list
│
├── results/
│   ├── results_loco_pooled_metrics.csv              Table 1
│   ├── results_loco_per_cave_metrics.csv            Per-cave LOCO breakdown
│   ├── results_within_cave_pooled_metrics.csv       Table 2
│   ├── results_within_cave_per_entity_metrics.csv   Per-entity WCBG breakdown
│   ├── local_baseline_comparison.csv                Model vs interpolation, per entity
│   ├── table2_linear_interp_pooled.csv              Pooled interpolation metrics
│   ├── li_rmse_iqr.csv                              Interpolation per-fold RMSE IQR
│   └── significance_tests.csv                       Wilcoxon, Bonferroni-adjusted
│
├── models/
│   └── *.joblib                          Serialized production pipelines
│
├── requirements.txt
└── README.md
```

---
## Web application

An interactive tool applying the trained models is available at:

**https://sisal-gap-filler.vercel.app/**

It accepts datasets in SISALv3 format for δ¹⁸O, δ¹³C, Mg/Ca and Sr/Ca, and provides model comparison and automated imputation. Serialized models are in `/models/`.

---
## Data

### Primary source

**SISALv3** — Speleothem Isotopes Synthesis and AnaLysis database, version 3.
Kaushal, N., Lechleitner, F. A., Wilhelm, M., et al. (2024). *Earth System Science Data* 16, 1933–1963. https://doi.org/10.5194/essd-16-1933-2024

The raw database is not redistributed here. The processed cohort used for all results is provided as `data/master_table_enriched.parquet`; the full database is publicly available from the link above.

### External enrichment

Two datasets are joined to each cave site by geographic coordinates:

- **WorldClim v2.1** — 19 bioclimatic variables at 1 km resolution (`wc_bio1`–`wc_bio19`). Fick, S. E. & Hijmans, R. J. (2017). *International Journal of Climatology* 37(12), 4302–4315.
- **Bowen–Wilkinson isoscape** — modelled mean annual precipitation δ¹⁸O at each site (`precip_d18o_bw02`). Bowen, G. J. & Wilkinson, B. (2002). *Geology* 30(4), 315.

WorldClim variables are modern climate normals (1970–2000) and therefore characterize relative climatic differences between cave locations rather than absolute paleoclimate values. They are joined at site level, so all samples from a given cave share the same covariate values.

### Cohort

The training cohort requires all four target proxies to be physically present in every retained sample. After this filter and a robust outlier screen:

- **18,004 samples**
- **41 cave sites**, latitude −35.7° to +54.2°
- **46 speleothem entities** meeting the ≥80-sample threshold for the within-cave protocol
- **36 input predictors** — 10 SISALv3 base variables, 6 engineered features, 19 WorldClim variables, 1 isoscape value

Sr isotope ratios are excluded (99.8% missing in the cohort).

The complete-case requirement restricts the cohort to a systematically well-measured subset of SISALv3. Records with sparser trace-element coverage — those for which imputation is most needed — are not represented. This is discussed as a limitation in the manuscript.

---

## Setup

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

Principal dependencies: `scikit-learn`, `xgboost`, `catboost`, `tensorflow`, `scikeras`, `pandas`, `numpy`, `scipy`, `pyarrow`, `ruptures`, `matplotlib`.

### Running from the processed data

The provided `data/master_table_enriched.parquet` is the assembled cohort. Evaluation and modelling cells can be run directly against it without obtaining the raw database.

### Running from raw SISALv3

To reproduce the assembly step, download SISALv3 and place the CSV export at:

```
<project_root>/data/sisalv3_database_mysql_csv/sisalv3_database_mysql_csv/sisalv3_csv/
```

Expected tables: `site`, `entity`, `sample`, `original_chronology`, `d18O`, `d13C`, `Mg_Ca`, `Sr_Ca`, `P_Ca`, `U_Ca`, `Ba_Ca`, `Sr_isotopes`, `hiatus`, `gap`.

The notebook resolves paths from a single project root, set by environment variable:

```bash
export SISAL_PROJECT_ROOT=/path/to/project
```

The pipeline was developed in Google Colab with the project directory mounted from Google Drive, and runs unmodified locally provided the root is set.

### Loading the serialized models

`CorrelationDropper` is a custom scikit-learn transformer used inside every production pipeline. **It must be defined in the session before any `.joblib` file is loaded**, as joblib does not persist class definitions. The definition is in the pipeline notebook.

---

## Methodological notes

### Leakage control

All preprocessing — median imputation, RobustScaler normalization, one-hot encoding of nominal features, and pairwise-correlation feature dropping at |r| ≥ 0.95 — is performed inside scikit-learn `Pipeline` and `ColumnTransformer` objects fitted on training folds only. No statistic derived from held-out samples influences the feature representation.

Random holdout splits are deliberately not used. Speleothem depth series are strongly autocorrelated, so a random split places samples from the same record on both sides of the partition, allowing a model to score well by identifying the record rather than by learning proxy–climate relationships. Both protocols hold out contiguous structure for this reason — entire cave sites under LOCO, contiguous depth blocks under WCBG.

### Target exclusion

All four target proxies are excluded from the predictor matrix for every target. No proxy is used to predict another. The non-target geochemical predictors are the trace-element ratios P/Ca, U/Ca and Ba/Ca.

### Chronology

Raw U–Th determinations and absolute interpolated age are excluded from the feature matrix, as dating uncertainties and post-publication age-model revisions vary systematically between records. Interpolated age enters only through a cyclical sine/cosine transform at the 21 kyr axial-precession period, which encodes orbital phase rather than absolute chronology.

### Outlier screening

A robust filter is applied to each target proxy within each entity: samples whose absolute deviation from the entity median exceeds three scaled median absolute deviations are removed. The scaling constant 1.4826 is applied, so the threshold corresponds to approximately 3σ under normality.

### Baselines

Two baselines are evaluated on the same folds as the models:

- **Linear interpolation**, using the measured samples bounding each held-out block as anchors. Applicable under WCBG only.
- **Ridge regression** on the climate covariates alone, under LOCO.

Differences between the tabular models and linear interpolation under WCBG are not statistically significant for any proxy (Wilcoxon signed-rank on per-entity RMSE, Bonferroni-adjusted for four comparisons, p > 0.9 in all cases). Under LOCO the tree ensembles outperform the ridge baseline on all four proxies.

### Prediction intervals

Two methods are used according to model family. Random Forest intervals derive from the dispersion of tree predictions (ensemble mean ± 1.96 × standard deviation across trees). CatBoost and XGBoost intervals derive from quantile regression at q = 0.025 and q = 0.975.

Intervals plotted in the manuscript figures are generated from production models fitted on the full cohort, whereas the prediction values derive from held-out fold predictions. Empirical coverage is reported in the manuscript and is below the nominal 95% for δ¹⁸O and δ¹³C.

---

## Reproducibility

A fixed seed (`random_state = 42`) is set throughout. Under a fixed seed and fixed thread count the tree-ensemble results are reproducible to the precision reported; small variation is possible across differing hardware or thread configurations owing to non-associative floating-point summation in parallel histogram construction. The deep learning models are subject to additional non-determinism from GPU kernel scheduling.

Linear interpolation has no stochastic component and returns identical values on every run given the same held-out blocks.

Full runtime for a complete pass is several hours, dominated by the deep learning models and the hyperparameter search.

---

## Web application

An interactive tool applying the trained models is available at:

**https://sisal-gap-filler.vercel.app/**

It accepts datasets in SISALv3 format for δ¹⁸O, δ¹³C, Mg/Ca and Sr/Ca, and provides model comparison and automated imputation.

---

## Supplementary data

Cave site details, hyperparameter configurations, training procedure documentation, and the processed dataset are archived at:

**https://doi.org/10.5522/04/31097830**

---

## Citation

> Altaweel, M., Khelifi, A., Balghouthi, M. R., Khelif, S., Paine, A., & Fleitmann, D. (submitted). *Bridging Gaps in Paleoclimatological Records: Insights from AI on Speleothem Data*. Ecological Informatics.

Please also cite SISALv3 (Kaushal et al., 2024), WorldClim v2.1 (Fick & Hijmans, 2017), and the Bowen–Wilkinson isoscape (Bowen & Wilkinson, 2002) as appropriate.

---

## License

Code is released under the MIT License. SISALv3 data are subject to the licence terms of the original database.

---

## Contact

Correspondence regarding the manuscript: **Mark Altaweel**, Institute of Archaeology, University College London — m.altaweel@ucl.ac.uk
