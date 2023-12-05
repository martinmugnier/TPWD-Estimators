% regularized nuclear norm estimator
function[beta] = nucnorm_reg(Y,X,lbda)
  [N, TT, K] = size(X);
  beta_init = rand(K,1);
  beta = fminsearch(@(beta) nucnorm_reg_obj(beta,Y, X,lbda), beta_init);
end

