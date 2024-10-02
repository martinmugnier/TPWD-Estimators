# README

This folder contains the MATLAB code for the paper "A Simple and Computationally Trivial Estimator for Grouped Fixed Effects Models" (Mugnier, 2022).

The code is made available to use at your own risk. If you use this code, please cite the Working Paper version available here: https://arxiv.org/abs/2203.08879.

For feedback or suggestions, feel free to contact me at: martin(dot)mugnier(at)psemail(dot)eu.

## Estimators

- **Triad pairwise-differencing (TPWD)**:
  - `TPWD_estimator.m`
  - `TPWD_estimator_without_covariates.m`

- **Nuclear norm**:
  - `MW_nucnorm_estimator.m`

- **Nuclear-norm regularized**:
  - `MW_nucnormreg_estimator_prox.m` (Algorithm 2)
  - `MW_nucnormreg_estimator_optim.m` (accelerated version described after Algorithm 2)

- **Grouped fixed-effects**:
  - `BM_algo1_multiple_init.m`
  - `BM_algo1_multiple_init_without_covariates.m`

- **Spectral**:
  - `CM_spectral_estimator.m`

- **Post-spectral**:
  - `CM_post_spectral_estimator.m`

- **Interactive fixed-effects least squares**:
  - `LS_factor.m`
  - `LS_factor_nnr_init.m`

## Replication Files

- **Monte Carlo simulations**:
  - `tables_1_and_2_and_S1_to_S6.m`
  - `tables_3_and_4_and_S7_to_S8.m`
  - `tables_S9_and_S10_unit_specific_effect.m`
  - `large_scale_simu.m`

- **Application**:
  - `application_income_democracy.m`
  - `export_acemoglu_data.do` (to generate the balanced sample)

- **Data**:
  - `5yearpanel.dta` (dowloaded from https://www.aeaweb.org/articles?id=10.1257/aer.98.3.808)
  - `acemoglu_balancedsample.csv`

### Replication Instructions:
To replicate the results of the paper:
1. Ensure all files, including `acemoglu_balancedsample.csv`, are in the same folder.
2. Run the relevant MATLAB scripts for each table (e.g., `tables_1_and_2_and_S1_to_S6.m` for Tables 1-2 and S1-6).

## Functions

### Functions for the TPWD estimator:
- **Step 2** (Sections 2.2 and 2.3):
  - `TPWD_clustering.m`
  - `TPWD_clustering_large.m`
  - `TPWD_clustering_squared.m`
  - `TPWD_clustering_large_squared.m`
  
- **Step 3** (Section 2.4):
  - `FE_reg_withcov.m`
  - `FE_reg_nocov.m`

### Functions for the grouped fixed-effects estimator:
  - `BM_algo1.m`
  - `reassign_groups.m`

### Functions for the post-spectral estimator:
  - `CM_classif.m`
  - `CM_spectral_clustering.m`

### Other functions:
- **Analytical standard errors**: `compute_GFE_analytical_cov.m`
- **Clustering accuracy**: `clustering_accuracy.m`
