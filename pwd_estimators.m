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
         
         function [G,grp_labels,grp_effects] = tpwd(Y,c)
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
             % take interactions
             % TBD
             % merge singleton groups
             % TBD
             % run pooled OLS and obtain group-specific effects estimates
             endog = reshape(Y',N*T,1);    
             %grp_effects = OLS(endog,exog);
             
             grp_effects = 0;
         end
         
         function [G,grp_labels,grp_effects,theta] = pwd2(Y,X,c,first_stage)
             % This function returns the PWD2 estimator.
             % INPUTS:
             % ------
             %         Y : NxT array of balanced panel data outcome; 
             %         X : NxTxp array of balanced panel data covariates;
             %         c : scalar threshold;
             %         first_stage : 0/1 dummy (first-diff./within).
             [N T p] = size(X); 
             % compute a first-stage estimate for theta
             if first_stage
                 Ydot = reshape((Y-mean(Y,2))',N*T,1);
                 Xdot = reshape(X-mean(X,2),N*T,p);
             else
                 Ydot = reshape(diff(Y,1,2)',N*(T-1),1);
                 Xdot = reshape(diff(X,1,2),N*(T-1),p);
             end
             theta1 = OLS(Ydot,Xdot);    
             % apply PWD1 to the residualized outcomes
             Ytilde = Y-X*theta1;
             [G,grp_labels,~] = pwd1(Ytilde,c);   
             % run pooled OLS and obtain the second-stage estimates
             exog = dummyvar(grp_labels);
             exog = repmat(exog',T,1);
             exog = reshape(exog,[],N*T)';
             endog = reshape(Y',N*T,1);
             exog = cat(2,exog,reshape(X,N*T,p));
             estimates = OLS(endog,exog);
             grp_effects = estimates(1:G);
             theta = estimates(G+1:end);
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
