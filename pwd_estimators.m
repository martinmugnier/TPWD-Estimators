 classdef pwd_estimators
     
     methods(Static)
         
         function [G,grp_labels,grp_effects] = pwd(Y,c)
             % This function returns the PWD estimator.
             % INPUTS: Y : NxT array of balanced panel data outcome; 
             %         c : scalar threshold.
             [N T] = size(Y);
             Ybar = nanmean(Y,2);   
             % compute the W matrix
             W = (bsxfun(@minus,Ybar,Ybar').^2<=c);     
             % obtain estimates for G and group labels
             G = size(unique(W,'rows'),1);
             [~,~,grp_labels] = unique(W,'rows'); 
             % generate group dummies
             exog = dummyvar(grp_labels); 
             exog = repmat(exog',T,1);
             exog = reshape(exog,[],N*T)';
             endog = reshape(Y',N*T,1);    
             % run pooled OLS and obtain group-specific effects estimates
             grp_effects = OLS(endog,exog);
         end
         
         function [G,grp_labels,params] = tpwd(Y,c)
             % This function returns the TPWD estimator.
             % INPUTS: Y : NxT array of balanced panel data outcome; 
             %         c : scalar threshold.
             [N T] = size(Y);
             s = permute(Y-permute(Y,[3 2 1]),[1 3 2]);
             S = max(abs(mean(s.*permute(s,[5 4 3 2 1]),3)),[],[4 5]);
             % compute the W matrix
             W = (S<=c);     
             % obtain estimates for G and group labels
             G = size(unique(W,'rows'),1);
             [~,~,grp_labels] = unique(W,'rows'); 
             % generate group dummies
             exog = dummyvar(grp_labels); 
             exog = repmat(exog',T,1);
             exog = reshape(exog,[],N*T)';
             timedum = kron(ones(N,1), eye(T));
             % take interactions exog x timedum
             % ...TBD...
             % merge singleton groups
             % ...TBD...
             % run pooled OLS and obtain structural parameters
             endog = reshape(Y',N*T,1);    
             params = OLS(endog,exog);
         end
       
    end
 end
 
 function[theta] = OLS(Y,X)
    % This function returns the OLS estimator.
    % INPUTS:
    % ------
    %         Y : n x 1 array; 
    %         X : n x p array.
    theta = inv(X'*X)*X'*Y;
 end
