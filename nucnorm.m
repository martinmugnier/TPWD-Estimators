% nuclear norm estimator
function [beta] = nucnorm(Y,X)
    [N, TT, K] = size(X);
    beta_init = rand(K,1);
    beta = fminsearch(@(beta) nucnorm_obj(beta, Y, X), beta_init);
end


