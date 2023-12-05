function [G,grp_labels,grp_effects] = tpwd_pureGFE(Y,c,clstr_algo)
     % This function returns the TPWD estimator for a model without 
     % covariates.
     % INPUTS: Y          : NxT array of possibly unbalanced panel data 
     %                      outcomes; 
     %         c          : nonnegative scalar tuning parameter;  
     %         clstr_algo : 0,1,2 select a specific agglomerative rule.
     [N,T] = size(Y);
     if sum(isnan(Y),'all')>0
         warning(['Missing values detected. A small value of ' ...
             'c may result in a lack of identification.']);
     end
     if any(sum(isnan(Y),2)>T-2)
         warning('Some units have less than two observations.');
     end
     % compute matrix of pairwise differenced outcomes at each time
     s = permute(Y-permute(Y,[3 2 1]),[1 3 2]);
     % compute matrix of pairwise distances using triads
     S = abs(mean(reshape(Y,N,1,1,T).*reshape(s,1,N,N,T),4, 'omitnan'));
     for j=1:N
         S(j,j,:) = 0;
         S(j,:,j) = 0;
     end
     S = squeeze(max(S,[],1));
     if clstr_algo==0
         % compute adjacency matrix
         W = (S<=c);     
         % group units with same test outcomes
         G = size(unique(W,'rows'),1);
         [~,~,grp_labels] = unique(W,'rows');
     elseif clstr_algo==1
         % group closest units in sequential order starting from 
         % the smallest  d(i,j) and make a cut after i when d(i,i+1)>c
         grp_labels=[];
         grp_labels(1,1) = 1;
         G = 1;
         clustered = [1];
         lasti = 1;
         k = 1;
         cont = true;
         while cont
             nonclustered = setdiff(1:N,clustered);
             [candval, newidx] = min(S(lasti,nonclustered));
             newi = nonclustered(newidx);
             if candval<=c
                 grp_labels(newi,1) = G;
             else
                 grp_labels(newi,1) = G+1;
                 G = G+1;
             end
             clustered(k+1,1) = newi;
             lasti = newi;
             k = k+1;
             cont = (k<N);
         end
     elseif clstr_algo==2
         % group closest units starting from the smallest  d(i,j)<=c
         % and merge all units within a weighted c distance to the
         % reference units
         grp_labels=[];
         G = 1;
         lowerS = tril(S,-1);
         lowerS(lowerS==0) = NaN;
         clustered = [];
         cont = true;
         while cont
             [m n] = min(lowerS(:),[], 'omitnan');
             [i j] = ind2sub(size(lowerS),n);
         if m>c
             grp_labels(setdiff(1:N, clustered),1) = G:G+N-length(clustered)-1;
             G = G+N-length(clustered)-1;
             cont = false;
         else %create a group containing only (i,j)
             clustered = [clustered i j];
             grp_labels([i j],1) = G;
             %add units within a c weighted-distance to (i,j) 
             to_merge = find(0.5*S(i,:)+0.5*S(j,:)<c);
             to_merge = setdiff(to_merge,clustered);
             grp_labels(to_merge, 1) = G;
             clustered = [clustered to_merge];
             %update remaning unmerged units
             lowerS(i,:) = NaN;
             lowerS(j,:) = NaN;
             lowerS(:,i) = NaN;
             lowerS(:,j) = NaN;
             lowerS(to_merge,:) = NaN;
             lowerS(:,to_merge) = NaN;
             if length(clustered)==N
                 cont = false;
             elseif length(clustered)==N-1
                 last_singleton_grp = setdiff(1:N,clustered);
                 grp_labels(last_singleton_grp,1) = G+1;
                 clustered = [clustered last_singleton_grp];
                 G = G+1;
                 cont = false;
             else
                 G = G+1;
             end
         end
         end
     end
     % run pooled OLS to obtain final estimates
     grp_effects = FE_reg_nocov(Y,grp_labels,true);
end