 classdef pwd_estimators
     
     methods(Static)
         
         function [G,grp_labels,grp_effects] = pwd(Y,c)
             % This function returns the PWD estimator.
             % INPUTS: Y : NxT array of possibly unbalanced
             %             panel data outcome; 
             %         c : scalar threshold.
             [N,T] = size(Y);
             if any(sum(isnan(Y),2)>T-2)
                 warning('Some units have less than two observations.');
             end
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
             nmissing = logical(1-isnan(endog));
             grp_effects = OLS(endog(nmissing,:),exog(nmissing,:));
         end
         
         function [G,grp_labels,grp_effects] = tpwd(Y,c)
             % This function returns the TPWD estimator.
             % INPUTS: Y        : NxT array of possibly unbalanced
             %                    panel data outcome; 
             %         c        : scalar threshold;           
             [N,T] = size(Y);
             if sum(isnan(Y),'all')>0
                 warning(['Missing values detected. A small value of c may result in a lack of identification.']);
             end
             if any(sum(isnan(Y),2)>T-2)
                 warning('Some units have less than two observations.');
             end
             s = permute(Y-permute(Y,[3 2 1]),[1 3 2]);
             S = max(abs(nanmean(s.*permute(s,[5 4 3 2 1]),3)),[],[4 5]);
             % compute the W matrix
             W = (S<=c);     
             % obtain estimates for G and group labels
             G = size(unique(W,'rows'),1);
             [~,~,grp_labels] = unique(W,'rows');
             % generate group dummies
             grp_dum = dummyvar(grp_labels); 
             grp_dum = repmat(grp_dum',T,1);
             grp_dum = reshape(grp_dum,[],N*T)';
             timedum = kron(ones(N,1), eye(T));
             % take interactions grp_dum x timedum
             exog =zeros(N*T,T*G);
             ii=0;
             for kk=1:G
                 for jj=1:T
                     ii = ii+1;
                     exog(:,ii) = grp_dum(:,kk).*timedum(:,jj);
                 end
             end
             % run pooled OLS and obtain group-specific effects estimates
             endog = reshape(Y',N*T,1);
             nmissing = logical((1-any(isnan(exog),2)).*(1-isnan(endog)));
             grp_effects = OLS(endog(nmissing,:),exog(nmissing,:));
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
 
