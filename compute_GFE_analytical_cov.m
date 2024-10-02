function[av_beta,av_alpha] = compute_GFE_analytical_cov(Y,X,beta,alpha,grp)
    % COMPUTE_GFE_ANALYTICAL_COV Return an estimator of the analytical 
    % standard error of the grouped fixed-effects (GFE) estimator of 
    % Bonhomme and Manresa (2015).
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS: 
    % -------
    % Y          : NxT array of balanced panel data outcomes;
    % X          : NxTxK array of balanced panel data covariates; 
    % beta       : Kx1 array of GFE common slope coefficient estimates;
    % alpha      : GxT array of GFE group-time fixed effect estimates;
    % grp        : Nx1 array of GFE group membership estimates.
    %
    % OUTPUTS:
    % --------
    % av_beta    : KxK array of asymptotic var-covar of the common slope;
    % av_alpha   : GxT array of standard erros of group-time fixed effects.
    %
    % REFERENCE:
    % ----------
    % Bonhomme, S. and Manresa, E. (2015), Grouped Patterns of 
    % Heterogeneity in Panel Data. Econometrica, 83: 1147-1184.
    
    [~,T,K] = size(X);
    Matgrp = dummyvar(grp);
    [N,G] = size(Matgrp);
    Xbar = zeros(G,T,K);
    res_X = zeros(N,T,K);
    for t=1:T
        Xbar(:,t,:) = (Matgrp'*Matgrp)\Matgrp'*reshape(X(:,t,:),N,K);
        res_X(:,t,:) = reshape(X(:,t,:),N,K)-Matgrp*reshape(Xbar(:,t,:),G,K);
    end
    residuals = Y-sum(X.*reshape(kron(beta',ones([N,T])),N,T,K),3)...
                     -Matgrp*alpha;
    Sigma = zeros([K K]);
    Omega = zeros([K K]);
    for i=1:N
        for t=1:T
            Sigma = Sigma + reshape(res_X(i,t,:),[K,1])*...
                reshape(res_X(i,t,:),[K,1])'./ (N*T);
            for s=1:T
                Omega = Omega + residuals(i,t).*residuals(i,s).*reshape...
                  (res_X(i,t,:),[K,1])*reshape(res_X(i,s,:),[K,1])'./(N*T);
            end
        end
    end
    Sigma_inv = inv(Sigma);
    av_beta = Sigma_inv*Omega*Sigma_inv./(N*T);
    av_alpha = zeros(G,T);
    for t=1:T
        av_alpha(:,t) = (Matgrp'*residuals(:,t)).^2./ ...
            (sum(Matgrp,1).^2)'./N;
    end
end
