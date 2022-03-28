 classdef pwd_estimators
     
     methods(Static)
         
         function [G,grp_labels,grp_effects] = pwd(Y,c)
             % This function returns the PWD estimator.
             % INPUTS: Y : NxT array of possibly unbalanced
             %             panel data outcome; 
             %         c : scalar threshold.
             [N,T] = size(Y);
             if any(sum(isnan(Y),2)>T-2)
                 warning('Some units have less than two observed outcomes');
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
         
         function [G,grp_labels,grp_effects] = tpwd(Y,c,merge,realloc)
             % This function returns the TPWD estimator.
             % INPUTS: Y        : NxT array of possibly unbalanced
             %                    panel data outcome; 
             %         c        : scalar threshold;
             %         merge    : if true -> merge groups until G<=N/2;
             %         realloc  : if true -> randomly reallocate units from
             %                    large groups (more than two members) to 
             %                    singleton groups (a unique members).            
             [N,T] = size(Y);
             if any(sum(isnan(Y),2)>T-2)
                 warning('Some units have less than two observed outcomes');
             end
             s = permute(Y-permute(Y,[3 2 1]),[1 3 2]);
             S = max(abs(nanmean(s.*permute(s,[5 4 3 2 1]),3)),[],[4 5]);
             % compute the W matrix
             W = (S<=c);     
             % obtain estimates for G and group labels
             G = size(unique(W,'rows'),1);
             [~,~,grp_labels] = unique(W,'rows');
             grp_effects = repelem(999,G*T);
             cont = true;
             if (N/G)<2
                 warning('G > N/2: cannot estimate group-specific effects.');
                 cont = false;
             end
             if merge
             % merge groups until G<=N/2
             % ...TBD...
             % cont = true
             end
             if cont*realloc  
                 % re-allocate units to singleton groups if needed and
                 % specified
                 deficient_group = [];
                 id_protected = [];
                 switched_id = [];
                 for j=1:G
                    if sum(grp_labels==j) <= 2
                        id_protected = [id_protected find(grp_labels==j)'];
                        if sum(grp_labels==j) < 2
                            deficient_group(end+1) = j;
                        end
                    end
                 end
                 if ~isempty(deficient_group)
                    warning('Singleton groups')
                    warning('Some units have been re-assigned to singleton groups.')
                    rng default
                    for j=deficient_group
                        id_not_yet_selected = true;
                        while id_not_yet_selected == true
                            selected_id = randi(N);
                            if sum([switched_id id_protected]==selected_id)==0
                                old_group = grp_labels(selected_id);
                                grp_labels(selected_id) = j;
                                id_not_yet_selected = false;
                                switched_id(end+1) = selected_id;
                                if sum(grp_labels==old_group) <= 2
                                    id_protected = [id_protected find(grp_labels==old_group)'];
                                end
                            end
                        end
                    end
                 end
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
                        ii=ii+1;
                        exog(:,ii)=grp_dum(:,kk).*timedum(:,jj);
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
