function[alpha,g] = ...
    BM_algo1_multiple_init_without_covariates(Y,G,n_init)
    % BM_ALGO1_MULTIPLE_INIT_WITHOUT_COVARIATES Return the grouped fixed- 
    % effects (GFE) estimator proposed in Bonhomme and Manresa (2015) for 
    % a model without covariates. It is obtained by selecting the best 
    % estimates (in terms of the final value of the objective function) 
    % obtained from `n_init' runs of 'Algorithm 1' starting at `n_init' 
    % random points in the parameter space chosen as in Section S.1.1 of 
    % Bonhomme and Manresa (2015)'s Supplementary Material. Without 
    % covariates, the implementation takes advantage of the fast MATLAB 
    % 'kmeans' routine.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu 
    % (last update: August 2024)
    %
    % INPUTS: 
    % -------
    % Y          : NxT array of balanced panel data outcomes;
    % G          : scalar number of groups;
    % n_init     : scalar number of starting points.
    %
    % OUTPUTS:
    % --------
    % alpha      : GxT array of group-time fixed effects;
    % g          : Nx1 array of group memberships.
    %
    % REFERENCE:
    % ----------
    % Bonhomme, S. and Manresa, E. (2015), Grouped Patterns of 
    % Heterogeneity in Panel Data. Econometrica, 83: 1147-1184.
    
    [N,T] = size(Y);
    if G>N 
        error(message('Number of groups greater than number of units.')); 
    end

    alpha_init = randn(G,T,n_init);
    for j=1:min(n_init,nchoosek(N,G))
        idx = randperm(N);
        alpha_init(:,:,j) = Y(idx(1:G),:);
    end
    [g,alpha] = kmeans(Y,G,'Start',alpha_init);
end


