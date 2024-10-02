function[beta] = MW_nucnormreg_estimator_optim(Y,X,psi)
    % MW_NUCNORMREG_ESTIMATOR_OPTIM Return the nuclear norm regularized 
    % estimator proposed in Moon and Weidner (2018). Computation
    % proceeeds by concentrating out the matrix of unobserved effects and
    % solving the concentrated problem using MATLAB's "fminsearch" 
    % built-in optimization routine.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y          : NxT array of balanced panel data outcome;
    % X          : NxTxK array of balanced panel data covariates;
    % psi        : scalar regularization parameter.
    %
    % OUTPUTS:
    % --------
    % beta       : Kx1 array of estimated common slope coefficients;
    %
    % REFERENCE:
    % ----------
    % Moon, H. R. and Weidner, M. (2018), Nuclear Norm Regularized 
    % Estimation of Panel Regression Models.

    [~,~,K] = size(X);
    beta_init = randn(K,1);
    beta = fminsearch(@(beta) nucnormreg_concentrated_obj(beta,Y,X,...
        psi),beta_init);
end


% regularized nuclear norm objective function
function [obj] = nucnormreg_concentrated_obj(beta,Y,X,psi)
  [N,TT,K] = size(X);
  single_index = sum(X.*reshape(kron(beta',ones([N,TT])),N,TT,K),3);
  obj = soft_thresholding((Y-single_index)/sqrt(N*TT),psi);
end

% useful functions
function [res] = q_psi(s,psi)
  res = s.^2./2.*(s<psi)+(psi.*s-psi^2/2).*(s>=psi);
end

function [res] = soft_thresholding(A,psi)
  SingVals = svd(A);
  res = sum(q_psi(SingVals,psi),'all');
end

