function [Ghat,g] = CM_spectral_clustering(Y,X,G,h,beta0,beta1,lbda_grid)
    % CM_SPECTRAL_CLUSTERING Return the spectral clustering algorithm 
    % proposed in Chetverikov and Manresa (2022), fine-tuning the lambda 
    % hyper-parameter across the grid 'lbda_grid'.
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
    % lbda_grid  : Jx1 array of classification hyper-parameters.
    %
    % OUTPUTS:
    % --------
    % G_hat      : estimated number of groups. In practice, CM recommend to 
    %              choose the minimal lambda in 'lbda_grid' such that 
    %              G_hat<=G;
    % g          : Nx1 array of estimated group memberships.
    %
    % REFERENCE:
    % ----------
    % Chetverikov, D. and Manresa, E. (2022), Spectral and post-spectral 
    % estimators for grouped panel data models.

    b = 0;
    stop = false;
    while ~stop
        Glist = zeros(length(lbda_grid),1);
        i = 1;
        for lbda = (lbda_grid+b)
            [Glist(i),~] = CM_classif(Y,X,G,h,beta0,beta1,lbda);
            i = i+1;
        end
        idx = Glist<=G;
        if sum(idx)==0
            %disp('There is no value of lambda which yields less than '+...
            %    string(G)+' groups.')
            b = b+max(lbda_grid)-min(lbda_grid)+1;
        else
            [~,i_star] = max(idx);
            stop = true;
        end
    end
    [Ghat,g] = CM_classif(Y,X,G,h,beta0,beta1,lbda_grid(i_star));
end