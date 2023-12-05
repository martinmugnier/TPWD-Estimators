function[beta, grp_effects] = FE_reg_withcov(Y,X,grp_labels,timevar)
    % This function returns the coefficients from an OLS regression of the 
    % outcome on the covariates and group (or group*time) dummies.
    %
    % INPUTS:
    % ------
    %         Y          : NxT array; 
    %         X          : NxK array;
    %         grp_labels : Nx1  array;
    %         timevar    : True/False.
    [N,T,K] = size(X);
    G = size(unique(grp_labels),1);
    grp_dum = dummyvar(grp_labels);
    grp_dum = repmat(grp_dum',T,1);
    grp_dum = reshape(grp_dum,[],N*T)';
    if timevar
        timedum = kron(ones(N,1),eye(T));
        % take interactions grp_dum x timedum
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
        grp_effects = reshape(estimates(1:G*T),T,G)';
        beta = estimates(G*T+1:end); 
    else
    grp_effects = estimates(1:G);
    beta = estimates(G+1:end);
    end
 end