%---------------- MONTE CARLO SIMULATIONS : large scale -------------------
% This code produces estimates for the large scale simulation described in 
% Section 4.2.
%
% Author: Martin Mugnier, Paris School of Economics
% Email: martin.mugnier (at) psemail (dot) eu 
% (last update: September 2024)
% 
%
% REFERENCE:
% ----------
% Mugnier (2022), A Simple and Computationally Trivial Estimator for 
% Grouped Fixed Effects Models.
%--------------------------------------------------------------------------

clear;

rng(202409);

% Settings
K = 1;
beta = ones(K,1);
balanced_grp = 1; 
error_correlation = 0;
sigma = 1/3; 

Gseq = [3,4];
Tseq = [7,10,20,40];

% BM (2015)'s GFE estimator
n_init_GFE = 100;
% Bai (2009)'s IFE estimator
n_init_IFE = 100; 
% Bonhomme and Manresa (2015)'s GFE estimator
Gmax = 5; 
% Chetverikov and Manresa (2021)'s spectral and post-spectral estimators
spectr_lbda_grid = linspace(1,50,100); 
M = 1; 
% Mugnier (2022)'s TPWD estimator
link = 'average'; 


N = 2000;
T = 7;
G = 4;
if balanced_grp==1
            grp = (1+(1:N>22)+(1:N>44)+(1:N>66))';
        else
           grp = (1+(1:N>2)+(1:N>N/10)+(1:N>N/2))';
end
grp_dum_mat = dummyvar(grp);
alpha = [];
alpha(1,:) = ones(1,T);
alpha(2,:) = ((1:T)-1)/(T-1);
alpha(3,:) = zeros(1,T);
split = floor(T/2);
alpha(4,:) = [zeros(1,split-1) ((split:T)-split)/(T-split)]; 
grp_effects_mat = grp_dum_mat*alpha;
if error_correlation==1
    noise_mod = arima('Constant',0,'AR',{0.20},...
                'Variance',sigma^2); 
    err = simulate(noise_mod,T,'NumPaths',N)'; 
else
    err = normrnd(0,sigma,N,T);
end
X = normrnd(0,sigma,N,T,K)+0.5*grp_effects_mat;
Y = sum(X.*reshape(kron(beta',ones([N,T])),N,T,K),3)+...
                grp_effects_mat+err;
            
% Step 1: Compute TPWD preliminary estimators
beta_nn = MW_nucnorm_estimator(Y,X);
psi_nnr = log(log(T))/(4*sqrt(min(N,T)));
tic
beta_nnr = MW_nucnormreg_estimator_optim(Y,X,psi_nnr); 
% Compute TPWD estimator (1 iteration)
% Step 2: Classification
residuals = Y-sum(X.*reshape(kron(beta_nnr',ones([N,T])),N,T,K),3);
sigma_hat = std(residuals,0,'all'); 
c = 1.5*sigma_hat^2*log(T)*T^(-1/2);
tic
[G_hatTPWD1,grp_hatTPWD1] = TPWD_clustering_large(residuals,c,link,false,true);
% Step 3: Linear projection
[beta_hatTPWD1,alpha_hatTPWD1] = FE_reg_withcov(Y,X,grp_hatTPWD1);
toc
[var_beta_hatTPWD1,~] = compute_GFE_analytical_cov(Y,X,...
      beta_hatTPWD1,alpha_hatTPWD1,grp_hatTPWD1);
 
 tic
% Compute TPWD estimator with 4 iterations
 beta_init = beta_hatTPWD1;
for j=1:3
    % Clustering
    residuals = Y-sum(X.*reshape(kron(...
          beta_init',ones([N,T])),N,T,K),3);
    sigma_hat = std(residuals,0,'all'); 
    c = 1.5*sigma_hat^2*log(T)*T^(-1/2);
    [G_hatTPWD2,grp_hatTPWD2] = TPWD_clustering_large(residuals,c,link,...
        false,true);
    % Linear projection
    [beta_hatTPWD2,alpha_hatTPWD2] = ...
          FE_reg_withcov(Y,X,grp_hatTPWD2);
    beta_init = beta_hatTPWD2;
end
[var_beta_hatTPWD2,~] = compute_GFE_analytical_cov(Y,X,...
     beta_hatTPWD2,alpha_hatTPWD2,grp_hatTPWD2);
toc

% Compute spectral and post-spectral estimators
tic
beta_hatS = CM_spectral_estimator(Y,X,G*M);
[beta_hatPS, alpha_hatPS,grp_hatPS] = CM_post_spectral_estimator(...
     Y,X,G,M,spectr_lbda_grid);
toc

% Compute GFE estimator (Algo 1)
tic
[beta_hatGFE, alpha_hatGFE,grp_hatGFE,objAlgo1] = ...
      BM_algo1_multiple_init(Y,X,G,n_init_GFE);
  toc
  [var_beta_hatGFE,~] = compute_GFE_analytical_cov(Y,X,...
     beta_hatGFE,alpha_hatGFE,grp_hatGFE);
 