function [Ghat,grp] = CM_classif(Y,X,G,h,beta0,beta1,lbda)
    % CM_CLASSIF Return group memberships estimated with the classification 
    % algorithm proposed in Chetverikov and Manresa (2022).
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
    % h          : Nx1 array of dummies for sample-splitting;
    % beta0      : Kx1 preliminary estimator computed on sample h==0;
    % beta1      : Kx1 preliminary estimator computed on sample h==1;
    % lbda       : regularization parameter for the number of groups.
    %
    % OUTPUTS:
    % --------
    % Ghat       : estimated number of groups. In practice, CM recommend to 
    %              choose the minimal lambda such that G_hat<=G;
    % grp        : Nx1 array of estimated group memberships.
    %
    % REFERENCE:
    % ----------
    % Chetverikov, D. and Manresa, E. (2022), Spectral and post-spectral 
    % estimators for grouped panel data models.

    
    [N,T,K]=size(X);
    n0 = sum(h==0);
    n1 = N-n0;
    res0 = Y(h==0,:)-sum(X(h==0,:,:).*reshape(kron(beta0',ones([n0,T])),...
        n0,T,K),3);
    res1 = Y(h==1,:)-sum(X(h==1,:,:).*reshape(kron(beta1',ones([n1,T])),...
        n1,T,K),3);
    B0 = reshape(res0,[n0,T,1]);
    B1 = reshape(res1,[n1,T,1]);
    B0 = squeeze(sum(bsxfun(@times,B0,permute(B0,[1,3,2])),1));
    B1 = squeeze(sum(bsxfun(@times,B1,permute(B1,[1,3,2])),1));
    [F0,~] = eigs(B0,G,'largestabs');
    [F1,~] = eigs(B1,G,'largestabs');
    A = zeros(N,T);
    A(h==0,:) = res0*(F0*F0');
    A(h==1,:) = res1*(F1*F1');
    grp = zeros(N,1);
    grp_mat = zeros(N,N);
    grp_mat(1,1) = 1;
    grp(1) = 1;
    m = 1;
    for i=2:N
        grp_candidate = [];
        for g=1:m
            if norm(A(i,:)-grp_mat(:,g)'*A./sum(grp_mat(:,g)))<=lbda
                grp_candidate = [grp_candidate,g];
            end
        end
        if sum(grp_candidate)==0
            m = m+1;
            grp(i) = m;
            grp_mat(i,m) = 1;
        else
            gamma = grp_candidate(1);
            grp_mat(i,gamma) = 1;
            grp(i) = gamma;
        end
    end
    Ghat = max(grp);
end
