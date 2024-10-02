function[alpha] = FE_reg_nocov(Y,g,timevar)
    % FE_REG_NOCOV Return the pooled panel OLS regression of the 
    % outcome on group*time (or group) dummies.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu 
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y          : NxT array of possibly unbalanced panel outcomes; 
    % g          : Nx1 array of group memberships;
    % timevar    : True (time-varying FE) / False (time-invariant FE).
    %
    % OUTPUT:
    % -------
    % alpha        : GxT array of group-time fixed effects.
    %
    % REFERENCE:
    % ----------
    % Mugnier (2022), A Simple and Computationally Trivial Estimator for 
    % Grouped Fixed Effects Models.
    
    if nargin<3
     timevar = true; %default choice is time-varying grouped fixed effects  
    end 
    
    [N,T] = size(Y);
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
    nmissing = logical((1-any(isnan(exog),2)).*(1-isnan(endog)));
    grp_effects = regress(endog(nmissing,:),exog(nmissing,:));
    if timevar
        alpha = reshape(grp_effects,T,G)';
    end
 end