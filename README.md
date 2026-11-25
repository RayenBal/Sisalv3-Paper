# Evaluating machine learning for gap filling in speleothem geochemical records

Code, data and results for:

> Altaweel, M., Khelifi, A., Balghouthi, M. R., Paine, A., Fleitmann, D., Alshloul, T., Toumi, A. (2026).
> *Evaluating machine learning for gap filling in speleothem geochemical records.* Applied Computing and Geosciences (under revision).

The study tests whether machine-learning models can impute missing δ¹⁸O, δ¹³C, Mg/Ca and Sr/Ca values in speleothem
records from the SISALv3 database (18,004 samples, 59 stalagmites, 41 caves). Five models are compared (Random Forest,
XGBoost, CatBoost, BiLSTM and Transformer) against linear interpolation and a ridge climate-covariate baseline, under
two leakage-free protocols:

- **Leave-one-cave-out (LOCO, 41 folds):** predicting a proxy at a cave the models have never seen.
- **Within-cave block gap (WCBG, 46 folds):** predicting a contiguous block withheld from a measured record, with the
  rest of the record available for training.

## Repository layout

```
notebooks/
  001_pipeline_train_evaluate.ipynb   training and evaluation pipeline (both protocols), baselines,
                                      statistical tests, export of the production models
  002_pipeline_evaluate.ipynb         evaluation pipeline: summary tables and figures from the saved runs
  revision/
    R1_reviewer_analyses.ipynb        gap duration, abrupt changes and hiatuses, prediction-interval
                                      coverage, representativeness of the cohort
    R2_feature_ablation.ipynb         tests of the age features and of the 21-kyr cyclical encoding
    R3_appendix_tables_figures.ipynb  Appendix A.3 tables and figures, revised Figure 6, consistency checks
    R4_reproducibility_check.ipynb    checks that the pipeline reproduces the 46 published within-cave blocks
data/
  master_table_enriched.parquet       modelling table (18,004 samples)
  cave_cohort_41sites.csv             metadata for the 41 cave sites
  pipeline_metadata.json              pipeline configuration, run identifiers and feature descriptions
  wcbg_heldout_blocks.csv             the 46 held-out blocks used for the within-cave evaluation (Table 2)
  runs/
    20260616T224121Z/                 leakage-free LOCO run: out-of-fold predictions, per-fold and pooled
                                      metrics, tuned hyper-parameters, configuration (Table 1)
    20260612T233845Z/                 leakage-free within-cave run: the same for the 46 held-out blocks (Table 2)
    revision_2026_11/                 outputs of the revision notebooks R1–R3 (block-level results,
                                      out-of-fold intervals, feature-ablation folds, appendix tables)
results/                              metrics behind Tables 1 and 2 (pooled and per cave / per record)
  revision/                           Appendix tables A.2–A.7 (CSV and Excel) and Figure 6 panel details
    checks/                           consistency checks (Table 2 interpolation row, model vs interpolation
                                      tests, tuned hyper-parameters in both runs)
figures/revision/                     Figure 6 and Appendix figures A.1–A.6 (PNG and PDF)
models/                               production models (joblib)
  hyperparameters/                    tuned hyper-parameters and run configuration (LOCO and within-cave runs)
```

## Running the notebooks

Open any notebook directly in Google Colab:

| Notebook | Open |
|---|---|
| 001 — training and evaluation pipeline | [Colab](https://colab.research.google.com/github/RayenBal/Sisalv3-Paper/blob/main/notebooks/001_pipeline_train_evaluate.ipynb) |
| 002 — evaluation, tables and figures | [Colab](https://colab.research.google.com/github/RayenBal/Sisalv3-Paper/blob/main/notebooks/002_pipeline_evaluate.ipynb) |
| R1 — reviewer analyses | [Colab](https://colab.research.google.com/github/RayenBal/Sisalv3-Paper/blob/main/notebooks/revision/R1_reviewer_analyses.ipynb) |
| R2 — feature ablation | [Colab](https://colab.research.google.com/github/RayenBal/Sisalv3-Paper/blob/main/notebooks/revision/R2_feature_ablation.ipynb) |
| R3 — Appendix A.3 tables and figures, Figure 6 | [Colab](https://colab.research.google.com/github/RayenBal/Sisalv3-Paper/blob/main/notebooks/revision/R3_appendix_tables_figures.ipynb) |
| R4 — reproducibility check of the within-cave blocks | [Colab](https://colab.research.google.com/github/RayenBal/Sisalv3-Paper/blob/main/notebooks/revision/R4_reproducibility_check.ipynb) |

The notebooks were run in Google Colab with the project folder on Google Drive (`MyDrive/SISALv3_Speleothem`); the
project root is set at the top of each notebook. To work from this repository, place its contents in that folder
layout: `data/master_table_enriched.parquet` → `artifacts/_data/`, and each folder in `data/runs/` →
`artifacts/<folder name>/`. With the saved runs in place, `002`, R3 and R4 recompute the reported numbers without
retraining, and R1 and R2 reuse the intermediate results saved in `data/runs/revision_2026_11/` (without them, R1 refits
the Random Forest folds and R2 retrains the best model per proxy for each feature variant, which takes several hours).
`001` retrains all models from the modelling table. The WorldClim v2.1 rasters used in R1 are downloaded from the
WorldClim website.

**Within-cave blocks.** The pipeline places one contiguous held-out block in each eligible record. To reproduce
Table 2 exactly, the notebooks read the published blocks from `data/wcbg_heldout_blocks.csv` (passed to
`BlockedWithinCaveSplitter(fixed_blocks=...)`). If the file is not found, blocks are placed with a fixed, session-
independent seed, following the same rule (10% of the record's samples, minimum 30), but at different positions.
`R4_reproducibility_check.ipynb` verifies that the published blocks are reproduced in all 12 saved prediction files.

**Hyper-parameters.** The tree ensembles were tuned with `HalvingRandomSearchCV` (20 candidates, three-fold
cave-grouped inner cross-validation); the deep-learning architectures were tuned on δ¹⁸O with the same inner
cross-validation and reused for the other proxies. The selected values are in `models/hyperparameters/`.

## Data sources

- SISALv3: Kaushal et al. (2024), *Earth System Science Data* 16, 1933–1963.
- WorldClim v2.1: Fick & Hijmans (2017), *International Journal of Climatology* 37, 4302–4315.
- Precipitation δ¹⁸O isoscape: Bowen & Wilkinson (2002), *Geology* 30, 315–318.

## Requirements

See `requirements.txt`. Results were produced with scikit-learn 1.6.1.
