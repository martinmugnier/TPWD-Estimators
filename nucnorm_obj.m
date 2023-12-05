% nuclear norm objective function
function [res] = nucnorm_obj(beta,Y,X)
  [N, TT, K] = size(X);
  single_index = sum(X.*reshape(kron(beta',ones([N,TT])),N,TT,K), 3);
  res = norm(svd(Y-single_index),1);
end