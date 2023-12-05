clear 

use "W:\Documents\MATLAB\code_for_a_simple_and_computationally_trivial_estimator_for_GFE_models\change_point_detection_algo\5yearpanel.dta"

* IDENTIFIERS (UNIT & PERIOD)
gen ID = code_numeric
gen TIME = year

* PANEL DATA SETUP
tsset ID TIME
sort ID TIME

* compute the covariates
gen lag_dem =  fhpolrigaug[_n-1] if ID[_n] == ID[_n-1]
gen lag_income =  lrgdpch[_n-1] if ID[_n] == ID[_n-1]

* keep only the relevant periods (1960-2000)
drop if year<1960

keep if samplebalancefe==1

keep  fhpolrigaug lag_dem lag_income 

export delimited using "W:\Documents\MATLAB\code_for_a_simple_and_computationally_trivial_estimator_for_GFE_models\change_point_detection_algo\acemoglu_balancedsample.csv", novarnames replace
