function [G,grp_labels,grp_effects] = pwd_pureGFE(Y,c)
% This function returns the PWD estimator without covariates.
% INPUTS: Y : N x T array of possibly unbalanced
%             panel data outcome;  
%         c : scalar threshold.
    [~,T] = size(Y);
    if any(sum(isnan(Y),2)>T-2)
        warning('Some units have less than two observations.');
    end
    Ybar = nanmean(Y,2);
    % compute adjacency matrix
    W = ((Ybar-Ybar').^2<=c);
    % estimate G and group labels
    G = size(unique(W,'rows'),1);
    [~,~,grp_labels] = unique(W,'rows');
    % run pooled OLS to obtain final estimates
    grp_effects = FE_reg(Y, grp_labels,false);
end