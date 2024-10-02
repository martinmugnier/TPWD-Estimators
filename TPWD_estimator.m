function [beta,alpha,g,obj] = TPWD_estimator(Y,X,psi,c,link,large)
    % This function returns the triad pairwise-differencing (TPWD) 
    % estimator proposed in Mugnier (2022).
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y          : NxT array of balanced panel data outcomes;
    % X          : NxTxK array of balanced panel data covariates; 
    % psi        : regularization parameter for the NNR estimator;
    % b_init     : initialization for computing the NNR estimator;
    % tol        : convergence tolerance for computing the NNR estimator;
    % c          : regularization parameter for TPWD clustering;
    % link       : linkage for the Hierarchichal Clustering Algorithm
    %              ('single', 'complete', 'average');
    % large      : dummy for the N*(N-1) loop version of the clustering 
    %              algorithm if N or T is large.
    %
    % OUTPUTS:
    % --------
    % beta       : Kx1 array of estimated common slope coefficients;
    % alpha      : GxT array of estimated group-time fixed effects;
    % g          : Nx1 array of estimated group memberships;
    % obj        : objective function value at the reported estimates.
    %
    % REFERENCE:
    % ----------
    % Mugnier, M. (2022), A Simple and Computationally Trivial Estimator 
    % for Grouped Fixed Effects Models.
    
    if nargin<4
        large = true; %default choice is large datasets  
    end 
    
    [N,T,K] = size(X);
    % Step 1: Preliminary nuclear-norm regularized estimator
    beta_nnr = nucnorm_reg_estimator(Y,X,psi); 
    % Step 2: Classification
    residuals = Y-sum(X.*reshape(kron(beta_nnr',ones([N,T])),N,T,K),3);
    if large
        [~,g] = TPWD_clustering_large(residuals,c,link);
    else
        [~,g] = TPWD_clustering(residuals,c,link);
    end
    % Step 3: Linear projection
    [beta,alpha] = FE_reg_withcov(Y,X,g,true);
    grp_dum_mat = dummyvar(g);
    obj = sum((Y-sum(X.*reshape(kron(beta',ones([N,T])),N,T,K),3)-...
        grp_dum_mat*alpha).^2,'all');
end