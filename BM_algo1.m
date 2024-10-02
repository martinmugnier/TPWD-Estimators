function [theta,alpha,g,obj] = BM_algo1(Y,X,G,max_iter,tol,theta_init,...
    alpha_init)
    % BM_ALGO1 Return the output of 'Algorithm 1' of Bonhomme and 
    % Manresa (2015).
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
    % max_iter   : maximum number of iterations;
    % tol        : tolerance for convergence;
    % theta_init : Kx1 array of common slope coefficients initializer;
    % alpha_init : GxT array of group-time fixed effects initializer.
    %
    % OUTPUTS:
    %---------
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
    % step 1: initial values
    theta = theta_init;
    alpha = alpha_init;
    g = zeros(N,1);
    for s = 1:max_iter 
        % step 2: assign groups
        for i = 1:N
            min_val = inf;
            for g_candidate = 1:G
                val = sum((Y(i,:)-squeeze(X(i,:,:))*theta-...
                    alpha(g_candidate,:)).^2);
                if val < min_val
                    min_val = val;
                    g(i) = g_candidate;
                end
            end
        end
        g = reassign_groups(g,G);  % ensure no group is empty
        % step 3: update theta and alpha using a pooled OLS regression
        theta_old = theta;
        alpha_old = alpha;
        [theta,alpha] = FE_reg_withcov(Y,X,g,true); 
        % check for convergence
        if max(abs(theta(:)-theta_old(:))) < tol && max(abs(alpha(:)-...
                alpha_old(:))) < tol
            break;
        end
    end
    grp_dum_mat = dummyvar(g);
    obj = sum((Y-sum(X.*reshape(kron(theta',ones([N,T])),N,T,K),3)-...
        grp_dum_mat*alpha).^2,'all');
end