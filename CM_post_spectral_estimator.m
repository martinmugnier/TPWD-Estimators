function [beta,alpha,g,obj] = CM_post_spectral_estimator(Y,X,G,M,lbda_grid)
    % CM_POST_SPECTRAL_ESTIMATOR Return the post-spectral estimator  
    % proposed in Chetverikov and Manresa (2022).
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y          : NxT array of balanced panel data outcome;
    % X          : NxTxK array of balanced panel data covariates; 
    % G          : number of groups in the outcome model;
    % M          : number of factors in the covariate model;
    % lbda_grid  : Jx1 array of classification hyper-parameters.
    %
    % OUTPUTS:
    % --------
    % beta       : Kx1 array of common slope coefficients;
    % alpha      : GxT array of group-time fixed effects;
    % g          : Nx1 array of group memberships;
    % obj        : objective function value at the reported estimates.
    %
    % REFERENCE:
    % ----------
    % Chetverikov, D. and Manresa, E. (2022), Spectral and post-spectral 
    % estimators for grouped panel data models.
    
    [N,T,K] = size(X);
    % Sample splitting 
    h = randi([0 1],N,1);
    % Step 1: Preliminary spectral estimator
    beta_s0 = CM_spectral_estimator(Y(h==0,:),X(h==0,:,:),G*M);
    beta_s1 = CM_spectral_estimator(Y(h==1,:),X(h==1,:,:),G*M);
    % Step 2: Classification
    [~,g] = CM_spectral_clustering(Y,X,G,h,beta_s0,beta_s1,lbda_grid);
    % Step 3: Linear projection
    [beta,alpha] = FE_reg_withcov(Y,X,g,true); 
    grp_dum_mat = dummyvar(g);
    obj = sum((Y-sum(X.*reshape(kron(beta',ones([N,T])),N,T,K),3)-...
        grp_dum_mat*alpha).^2,'all');
end