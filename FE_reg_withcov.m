function[beta,alpha] = FE_reg_withcov(Y,X,g,timevar)
    % FE_REG_WITHCOV Return the coefficients from an OLS regression of the 
    % outcome on the covariates and group (or group*time) dummies.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y            : NxT array of balanced panel data outcome;
    % X            : NxTxK array of balanced panel data covariates;
    % g            : Nx1 array of group memberships;
    % timevar      : include interactions with time dummies if true.
    %
    % OUTPUT:
    % -------
    % alpha        : GxT array of group-time fixed effects.
    %
    % REFERENCE:
    % ----------
    % Mugnier (2022), A Simple and Computationally Trivial Estimator for 
    % Grouped Fixed Effects Models.
    
    if nargin<4
     timevar = true; %default choice is time-varying grouped fixed effects  
    end 
	
    [N,T,K] = size(X);
    G = size(unique(g),1);
    grp_dum = dummyvar(g);
    grp_dum = repmat(grp_dum',T,1);
    grp_dum = reshape(grp_dum,[],N*T)';
    if timevar
        timedum = kron(ones(N,1),eye(T));
        % take interactions grp_dum*timedum
        exog = zeros(N*T,T*G);
        ii = 0;
        for kk = 1:G
            for jj = 1:T
                ii = ii+1;
                exog(:,ii) = grp_dum(:,kk).*timedum(:,jj);
            end
        end
    else
        exog = grp_dum;
    end
    % run pooled OLS and obtain final estimates
    endog = reshape(Y',N*T,1);
    for k = 1:K
        exog = cat(2,exog,reshape(X(:,:,k)',N*T,1));
    end
    nmissing = logical((1-any(isnan(exog),2)).*(1-isnan(endog)));
    estimates = regress(endog(nmissing,:),exog(nmissing,:));
    if timevar
        alpha = reshape(estimates(1:G*T),T,G)';
        beta = reshape(estimates(G*T+1:end),K,1); 
    else
        alpha = reshape(estimates(1:G),G,1);
        beta = reshape(estimates(G+1:end),K,1);
    end
 end