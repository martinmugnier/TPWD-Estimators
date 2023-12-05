**READ ME for the code associated with ``A Simple and Computationally Trivial Estimator for Grouped Fixed Effects Models'' (Mugnier, Working Paper, 2022).**

This folder contains MATLAB code for 

   - Step 1 of the TPWD estimator using the regularized nuclear norm estimator described in Section 2.1 or the nuclear norm estimator of Moon and Weidner (2019):  `nucnorm_reg_obj.m`, `nucnorm_reg.m’, `nucnorm_obj.m’, and `nucnorm.m’.
   - Step 2 of the TPWD estimator as described in Sections 2.2 and 2.3: `tpwd_pureGFE.m’.
   - Step 3 of the TPWD estimator as described in Section 2.4: `FE_reg_withcov.m’ and `FE_reg_nocov.m’.
   - Computing analytical standard errors based on large N,T estimation of the asymptotic variance and estimated groups: `compute_GFE_analytical_cov.m’.
   - Assessing clustering accuracy in Monte Carlo experiments: `clustering_accuracy.m’.
   - Computing an approximation of Bonhomme and Manresa (2015)'s GFE estimator without covariates based on Lloyd's algorithm: `GFE.m’.

This code is used by  `tables_1_and_2_and_S1_to_S6.m', `tables_3_and_4.m', `tables_S7_and_S8_unit_specific_effect.m', and  `tables_S9_and_S10_lagged_outcome.m' to produce the Monte Carlo results (Tables 1-4 and S1-10), and by `application_income_democracy.m' to produce the application results (Table 5 and Figures 1 and 2). Tables S12-14 and Figures S1-6 can be obtained by a simple adaptation of the code.

To replicate the results of the paper, first ensure all files are located in the same folder as well as the data `acemoglu_balancedsample.csv'. This data is obtained after downloading `5yearpanel.dta' from `https://www.aeaweb.org/articles?id=10.1257/aer.98.3.808.' and running the STATA code `export_acemoglu_data.do'. 

The code is made available to use at your own risk (please cite the Working Paper version available here: https://arxiv.org/pdf/2203.08879.pdf).

If you have any feedback/improvement suggestions, please feel free to reach me at: martin.mugnier@economics.ox.ac.uk.
