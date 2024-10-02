function [beta] = MW_nucnorm_estimator(Y,X)
    % MW_NUCNORM_ESTIMATOR Return the nuclear norm estimator proposed in 
    % Moon and Weidner (2018). Computation uses MATLAB's "fminsearch" 
    % built-in optimization routine.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu 
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y          : NxT array of balanced panel data outcome;
    % X          : NxTxK array of balanced panel data covariates.
    %
    % OUTPUT:
    % -------
    % beta       : Kx1 array of estimated common slope coefficients.
    %
    % REFERENCE:
    % ----------
    % Moon, H. R. and Weidner, M. (2018), Nuclear Norm Regularized 
    % Estimation of Panel Regression Models.

    [~,~,K] = size(X);
    beta_init = randn(K,1);
    beta = fminsearch(@(beta) nucnorm_obj(beta,Y,X),beta_init);
end


% nuclear norm objective function
function [res] = nucnorm_obj(beta,Y,X)
  [N,TT,K] = size(X);
  single_index = sum(X.*reshape(kron(beta',ones([N,TT])),N,TT,K),3);
  res = norm(svd(Y-single_index),1);
end

