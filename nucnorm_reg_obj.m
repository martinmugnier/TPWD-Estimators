% regularized nuclear norm objective function
function [obj] = nucnorm_reg_obj(beta,Y,X,lbda)
  [N,TT,K] = size(X);
  single_index = sum(X.*reshape(kron(beta',ones([N,TT])),N,TT,K), 3);
  obj = soft_thresholding((Y-single_index)/sqrt(N*TT), lbda);
end

% useful functions
function [res] = q_lbda(s,lbda)
  res = s.^2./2.*(s<lbda)+(lbda.*s-lbda^2/2).*(s>=lbda);
end

function [res] = soft_thresholding(A,lbda)
  SingVals = svd(A);
  res = sum(q_lbda(SingVals,lbda), 'all');
end




