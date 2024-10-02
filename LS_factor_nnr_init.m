function beta = LS_factor_nnr_init(Y,X,R,psi)
    % This function returns Chetverikov and Manresa (2022)'s definition
    % of Moon and Weidner (2018)'s nuclear-norm regularized estimator.
    % The only difference is that psi will be chosen according to the theory
    % developed in MW2018 given that the procedure does not generate any
    % group so that it seems the implementation described in CM2022 is not 
    % feasible.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y           : NxT array of balanced panel data outcome;
    % X           : NxTxK array of balanced panel data covariates; 
    % R           : rank of the factor model (typically equal to the number
    %               of groups in the GFE outcome model);
    % psi         : regularization parameter.
    %
    % OUTPUT:
    % -------
    % beta        : Kx1 array of common slope coefficients.
    %
    % REFERENCES:
    % -----------
    % Bai, J. (2009), Panel Data Models With Interactive Fixed Effects. 
    % Econometrica, 77: 1229-1279.
    % Moon, H. R. and Weidner, M. (2018), Nuclear Norm Regularized 
    % Estimation of Panel Regression Models.
    % Chetverikov, D. and Manresa, E. (2022), Spectral and post-spectral 
    % estimators for grouped panel data models.
    
    [N,T,K] = size(X);
    beta_nnr = MW_nucnormreg_estimator(Y,X,psi,zeros(K,1),1e-5);
    XX = zeros(K,N,T);
    for k=1:K
        XX(k,:,:) = X(:,:,k);
    end
    beta = LS_factor(Y,XX,R,'silent',1e-8,'m1',beta_nnr,1);
end

