function[grp_effects] = FE_reg_nocov(Y,grp_labels,timevar)
    % This function returns the OLS regression of the outcome on 
    % group OR group*time dummies.
    % INPUTS:
    % ------
    %         Y          : NxT array; 
    %         grp_labels : Nx1  array;
    %         timevar    : True/False.
    [N,T] = size(Y);
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
    nmissing = logical((1-any(isnan(exog),2)).*(1-isnan(endog)));
    grp_effects = regress(endog(nmissing,:),exog(nmissing,:));
    if timevar
        grp_effects = reshape(grp_effects,T,G)';
    end
 end