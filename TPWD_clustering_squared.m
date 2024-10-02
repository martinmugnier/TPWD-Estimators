function [G,g] = TPWD_clustering_squared(Y,c)
     % TPWD_CLUSTERING Return the triad pairwise-differencing (TPWD) 
     % clustering estimator proposed in Mugnier (2022), with squared
     % distance.
     %
     % Author: Martin Mugnier
     % Email: martin.mugnier (at) psemail (dot) eu
     % (last update: August 2024)
     %
     % INPUTS:
     % -------
     % Y          : NxT array of possibly unbalanced panel data outcomes; 
     % c          : nonnegative scalar thresholding parameter;  
     %
     % OUTPUTS:
     % --------
     % G          : scalar estimated number of groups;
     % g          : Nx1 array of estimated group memberships.
     %
     % REFERENCE:
     % ----------
     % Mugnier, M. (2022), A Simple and Computationally Trivial Estimator 
     % for Grouped Fixed Effects Models.
     
     [N,T] = size(Y);
     if sum(isnan(Y),'all')>0
         warning(['Missing values detected. A small value of ' ...
             'c may result in a lack of identification.']);
     end
     if any(sum(isnan(Y),2)>T-2)
         warning('Some units have less than two observations.');
     end
     % compute matrix of pairwise outcome differences at each time period
     s = permute(Y-permute(Y,[3 2 1]),[1 3 2]);
     % compute dissimilarity matrix using pairwise differences and triads 
     D = abs(mean(reshape(Y,N,1,1,T).*reshape(s,1,N,N,T),4,...
         'omitnan'));
     for j=1:N
         D(j,j,:) = 0;
         D(j,:,j) = 0;
     end
     D = squareform(squeeze(max(D,[],1))).^2;
     % perform agglomerative clustering
     W = linkage(D,'average');
     g = cluster(W,'cutoff',c,'Criterion','distance');
     G = length(unique(g));
end