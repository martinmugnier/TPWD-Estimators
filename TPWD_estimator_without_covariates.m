function [G,alpha,g,obj] = TPWD_estimator_without_covariates(Y,c,link,large)
    % TPWD_ESTIMATOR_WITHOUT_COVARIATES Return the triad 
    % pairwise-differencing (TPWD) estimator proposed in Mugnier (2022) for 
    % a model without covariates.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y          : NxT array of balanced panel data outcomes;
    % c          : regularization parameter for TPWD clustering;
    % link       : linkage for the Hierarchichal Clustering Algorithm
    %              ('single', 'complete', 'average');
    % large      : dummy for the light-memory N*(N-1) loop version of the 
    %              clustering algorithm if N or T is large.
    %
    % OUTPUTS:
    % --------
    % G          : scalar estimated number of groups;
    % alpha      : GxT array of estimated group-time fixed effects;
    % g          : Nx1 array of estimated group memberships;
    % obj        : objective function value at the reported estimates.
    %
    % REFERENCE:
    % ----------
    % Mugnier, M. (2022), A Simple and Computationally Trivial Estimator 
    % for Grouped Fixed Effects Models.
    
    if nargin<3
        link = 'average'; %default choice is large datasets  
    end
    if nargin<4
        large = true; %default choice is large datasets  
    end 
    % Step 1: Classification
    if large
        [G,g] = TPWD_clustering_large(Y,c,link);
    else
        [G,g] = TPWD_clustering(Y,c,link);
    end
    % Step 2: Linear projection
    alpha = FE_reg_nocov(Y,g); 
    grp_dum_mat = dummyvar(g);
    obj = mean((Y-grp_dum_mat*alpha).^2,'all');
end