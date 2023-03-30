        
 function [G,grp_labels,grp_effects] = tpwd_pureGFE(Y,c)
     % This function returns the TPWD estimator.
     % INPUTS: Y        : NxT array of possibly unbalanced
     %                    panel data outcome; 
     %         c        : scalar threshold;           
     [N,T] = size(Y);
     if sum(isnan(Y),'all')>0
         warning(['Missing values detected. A small value of ' ...
             'c may result in a lack of identification.']);
     end
     if any(sum(isnan(Y),2)>T-2)
         warning('Some units have less than two observations.');
     end
     s = permute(Y-permute(Y,[3 2 1]),[1 3 2]);
     % compute test statistics using tetrad (computationally intensive)
     %S = max(abs(nanmean(s.*permute(s,[5 4 3 2 1]),3)),[],[4 5]);
     % compute test statistics using triads (less computationally
     % intensive)
     S = squeeze(max(abs(nanmean(reshape(Y,N,1,1,T).*...
         reshape(s,1,N,N,T),4)),[],1));
     % compute adjacency matrix
     W = (S<=c);     
     % estimate G and group labels
     G = size(unique(W,'rows'),1);
     [~,~,grp_labels] = unique(W,'rows');
     % run pooled OLS to obtain final estimates
     grp_effects = FE_reg(Y,grp_labels,true);
 end