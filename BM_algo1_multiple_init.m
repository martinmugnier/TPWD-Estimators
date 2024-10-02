function [theta,alpha,g,obj] = BM_algo1_multiple_init(Y,X,G,n_init)
    % BM_ALGO1_MULTIPLE_INIT Return the grouped fixed-effects (GFE) 
    % estimator proposed in Bonhomme and Manresa (2015) and obtained from 
    % selecting the best estimates (in terms of the final value of the 
    % objective function) obtained from `n_init' runs of their 'Algorithm 1' 
    % starting at `n_init' random points in the parameter space chosen as 
    % in Section S.1.1 of Bonhomme and Manresa (2015)'s Supplementary 
    % Material.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu 
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y          : NxT array of balanced panel data outcome;
    % X          : NxTxK array of balanced panel data covariates; 
    % G          : number of groups;
    % n_init     : number of starting points.
    %
    % OUTPUTS:
    % --------
    % theta      : Kx1 array of common slope coefficients;
    % alpha      : GxT array of group-time fixed effects;
    % g          : Nx1 array of group memberships;
    % obj        : objective function value at the reported estimates.
    %
    % REFERENCE:
    % ----------
    % Bonhomme, S. and Manresa, E. (2015), Grouped Patterns of 
    % Heterogeneity in Panel Data. Econometrica, 83: 1147-1184.
    
    [N,T,K] = size(X);
    min_obj = inf;
    for s = 1:n_init
        theta_init = randn(K,1);
        rand_units = randsample(N,G);
        alpha_init = Y(rand_units,:)-sum(X(rand_units,:,:).*reshape(...
            kron(theta_init',ones([G,T])),G,T,K),3);
        [theta_temp,alpha_temp,g_temp,obj_temp] = BM_algo1(Y,X,G,100,...
            1e-5,theta_init,alpha_init);
        if obj_temp < min_obj
            min_obj = obj_temp;
            theta = theta_temp;
            alpha = alpha_temp;
            g = g_temp;
            obj = obj_temp;
        end
    end